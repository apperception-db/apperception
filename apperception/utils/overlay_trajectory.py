from pickle import FRAME
from apperception.database import database
from apperception.utils import transformation
# from apperception.world import World
import cv2
import numpy as np
from typing import (TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set,
                    Tuple, Union)
import datetime
FRAME_RATE = 10
FONT = cv2.FONT_HERSHEY_SIMPLEX



def overlay_trajectory(
    world,
    images_data_path: str = None,
    num_joined_tables: int = 1,
    is_overlay_headings: bool = False,
    is_overlay_road: bool = False,
    is_overlay_objects: bool = False,
    is_keep_whole_video: bool = False
):
    id_time_camId_filename = world.get_id_time_camId_filename(num_joined_tables=num_joined_tables)
    frames = {}
    itemIds = set()
    for frame in id_time_camId_filename:
        objId, time, camId, filename = frame
        itemIds.add(objId)
        file_prefix = "-".join(filename.split("/")[:-1])
        if (file_prefix, camId) not in frames:
            frames[(file_prefix, camId)] = []
        frames[(file_prefix, camId)].append(frame)

    for (file_prefix, camId) in frames:
        frames[(file_prefix, camId)].sort(key=lambda x: x[1])  # sort base on time
        frame_width = None
        frame_height = None
        vid_writer = None
        for frame in frames[(file_prefix, camId)]:
            objId, time, camId, cam_filename = frame
            filename = cam_filename
            if images_data_path is not None:
                filename_no_prefix = filename.split("/")[-1]
                filename = images_data_path + "/" + filename_no_prefix
            frame_im = cv2.imread(filename)
            
            camera_config = fetch_camera_config(cam_filename)
            camera_config["time"] = time
            if is_overlay_objects:
                frame_im = overlay_objects(frame_im, itemIds, camera_config)
            if is_overlay_headings:
                frame_im = overlay_stats(frame_im, camera_config)
            if is_overlay_road:
                frame_im = overlay_road(world, frame_im)

            if vid_writer is None:
                frame_height, frame_width = frame_im.shape[:2]
                vid_writer = cv2.VideoWriter(
                    "./output/" + file_prefix + "." + camId + ".mp4",
                    cv2.VideoWriter_fourcc("m", "p", "4", "v"),
                    FRAME_RATE,
                    (frame_width, frame_height),
                )
            vid_writer.write(frame_im)
        if vid_writer is not None:
            vid_writer.release()

##### SQL Utils #####
def fetch_camera_config(filename: str):
    query = f"""
    CREATE OR REPLACE FUNCTION ST_XYZ (g geometry) RETURNS real[] AS $$
        BEGIN
            RETURN ARRAY[ST_X(g), ST_Y(g), ST_Z(g)];
        END;
    $$ LANGUAGE plpgsql;

    SELECT
        cameraId,
        ST_XYZ(egoTranslation),
        egoRotation,
        ST_XYZ(cameraTranslation),
        ST_XYZ(cameraTranslationAbs),
        cameraRotation,
        cameraIntrinsic,
        frameNum,
        fileName,
        cameraHeading,
        egoHeading
    FROM Cameras
    WHERE
        fileName = '{filename}'
    ORDER BY cameraId ASC, frameNum ASC;
    """
    # print(query)
    result = database._execute_query(query)[0]
    print(result)
    camera_config = {
        "cameraId": result[0],
        "egoTranslation": result[1],
        "egoRotation": result[2],
        "cameraTranslation": result[3],
        "cameraTranslationAbs": result[4],
        "cameraRotation": result[5],
        "cameraIntrinsic": result[6],
        "frameNum": result[7],
        "fileName": result[8],
        "cameraHeading": result[9],
        "egoHeading": result[10],
    }
    return camera_config

def fetch_trajectory(itemId: str):
    query = f"""
            SELECT asMFJSON(trajCentroids)::json->'sequences'
            FROM Item_General_Trajectory as final
            WHERE itemId = '{itemId}'
            """

    traj = database._execute_query(query)[0][0][0]
    coordinates = traj["coordinates"]
    datetimes = traj["datetimes"]
    result = {}
    for coord, time in zip(coordinates, datetimes):
        dt_time = datetime.datetime.strptime(time, "%Y-%m-%dT%H:%M:%S.%f+00").replace(tzinfo=datetime.timezone.utc)
        result[dt_time] = coord
    return result



##### CV2 Overlay Utils #####
def overlay_objects(frame, itemIds, camera_config):
    time = camera_config["time"]
    pixels = {}
    for itemId in itemIds:
        traj = fetch_trajectory(itemId=itemId)
        if time in traj:
            current_traj_point = traj[time]
            
            current_pixel = world_to_pixel(camera_config, current_traj_point)

            pixels[itemId] = current_pixel

    print(pixels)
    for itemId in pixels:
        pixel = pixels[itemId]
        cv2.circle(
            frame,
            tuple([int(pixel[0][0]), int(pixel[1][0])]),
            10,
            (0, 255, 0),
            -1,
        )
    return frame


def overlay_stats(frame, camera_config):
    ego_translation = camera_config["egoTranslation"]
    ego_heading = camera_config["egoHeading"]
    camera_heading = camera_config["cameraHeading"]
    cam_road_dir = database.road_direction(ego_translation[0], ego_translation[1], ego_heading)[0][0]
    
    stats = {
        "Ego Heading": str(round(ego_heading, 2)),
        "Camera Heading": str(round(camera_heading, 2)),
        "Road Direction": str(round(cam_road_dir, 2)),
    }
    
    x, y = 10, 50
    for stat in stats:
        statValue = stats[stat]
        cv2.putText(
            frame,
            stat + ": " + statValue,
            (x, y),
            FONT,
            1,
            (0, 255, 0),
            3,
        )
        y += 50
    return frame

def overlay_road(frame, camera_config):
    ego_translation = camera_config["egoTranslation"]
    camera_road_coords = database.road_coords(ego_translation[0], ego_translation[1])[0][0]
    pixel_start_enc = world_to_pixel(
        camera_config, (camera_road_coords[0], camera_road_coords[1], 0)
    )
    pixel_end_enc = world_to_pixel(
        camera_config, (camera_road_coords[2], camera_road_coords[3], 0)
    )
    pixel_start = tuple([int(pixel_start_enc[0][0]), int(pixel_start_enc[1][0])])
    pixel_end = tuple([int(pixel_end_enc[0][0]), int(pixel_end_enc[1][0])])
    print(camera_road_coords, ego_translation)
    frame = cv2.line(
        frame,
        pixel_start,
        pixel_end,
        (255, 0, 0),
        3,
    )
    return frame


##### Camera Transformation Utils #####
def world_to_pixel(
    camera_config: dict, world_coords: Union[np.ndarray, Tuple[float, float, float]]
):
    traj_2d = transformation(world_coords, camera_config)
    return traj_2d


# def overlay_trajectory(
#     overlay_headings: bool = False,
#     overlay_road: bool = False,
#     overlay_objects: bool = False,
#     keep_whole_video: bool = False
# ):
#     frame_nums = database.timestamp_to_framenum(
#         scene_name, ["'" + x + "'" for x in trajectory.datetimes]
#     )
#     # c amera_info is a list of list of cameras, where the list of cameras at each index represents the cameras at the respective timestamp
#     camera_info: List[List["FetchCameraTuple"]] = []
#     for frame_num in frame_nums:
#         current_cameras = database.fetch_camera_framenum(scene_name, [frame_num[0]])
#         camera_info.append(current_cameras)

#     overlay_info = get_overlay_info(trajectory, camera_info)

#     for file_prefix in overlay_info:
#         frame_width = None
#         frame_height = None
#         vid_writer = None
#         camera_points = overlay_info[file_prefix]
#         for point in camera_points:
#             (
#                 traj_2d,
#                 framenum,
#                 filename,
#                 camera_heading,
#                 ego_heading,
#                 ego_translation,
#                 camera_config,
#             ) = point
#             frame_im = cv2.imread(filename)
#             cv2.circle(
#                 frame_im,
#                 tuple([int(traj_2d[0][0]), int(traj_2d[1][0])]),
#                 10,
#                 (0, 255, 0),
#                 -1,
#             )
#             if overlay_headings:
#                 cam_road_dir = self.road_direction(ego_translation[0], ego_translation[1])[0][0]
#                 stats = {
#                     "Ego Heading": str(round(ego_heading, 2)),
#                     "Camera Heading": str(round(camera_heading, 2)),
#                     "Road Direction": str(round(cam_road_dir, 2)),
#                 }
#                 self.overlay_stats(frame_im, stats)
#             if overlay_road:
#                 frame_im = self.overlay_road(frame_im, camera_config)
#             if vid_writer is None:
#                 frame_height, frame_width = frame_im.shape[:2]
#                 vid_writer = cv2.VideoWriter(
#                     "./output/" + file_name + "." + file_prefix + ".mp4",
#                     cv2.VideoWriter_fourcc("m", "p", "4", "v"),
#                     10,
#                     (frame_width, frame_height),
#                 )
#             vid_writer.write(frame_im)
#         if vid_writer is not None:
#             vid_writer.release()

##### General Utils #####
# def get_overlay_info(trajectory: Trajectory, camera_info: List[List["FetchCameraTuple"]]):
#     """
#     overlay each trajectory 3d coordinate on to the frame specified by the camera_info
#     1. For each point in the trajectory, find the list of cameras that correspond to that timestamp
#     2. Project the trajectory coordinates onto the intrinsics of the camera, and add it to the list of results
#     3. Returns a mapping from each camera type (FRONT, BACK, etc) to the trajectory in pixel coordinates of that camera
#     """
#     traj_obj_3d = trajectory.coordinates
#     result: Dict[str, List[Tuple[np.ndarray, int, str, float, float, List[float], dict]]] = {}
#     for index, cur_camera_infos in enumerate(camera_info):
#         # cur_camera_infos = camera_info[
#         #     index
#         # ]  # camera info of the obejct in one point of the trajectory
#         centroid_3d = np.array(traj_obj_3d[index])  # one point of the trajectory in 3d
#         # in order to fit into the function transformation, we develop a dictionary called camera_config
#         for cur_camera_info in cur_camera_infos:
#             # TODO: add type to camera_config
#             camera_config: Dict[str, Any] = {}
#             camera_config["egoTranslation"] = cur_camera_info[1]
#             camera_config["egoRotation"] = np.array(cur_camera_info[2])
#             camera_config["cameraTranslation"] = cur_camera_info[3]
#             camera_config["cameraRotation"] = np.array(cur_camera_info[4])
#             camera_config["cameraIntrinsic"] = np.array(cur_camera_info[5])

#             traj_2d = world_to_pixel(camera_config, centroid_3d)

#             framenum = cur_camera_info[6]
#             filename = cur_camera_info[7]
#             camera_heading = cur_camera_info[8]
#             ego_heading = cur_camera_info[9]
#             ego_translation = cur_camera_info[1]
#             file_prefix = "_".join(filename.split("/")[:-1])

#             if file_prefix not in result:
#                 result[file_prefix] = []
#             result[file_prefix].append(
#                 (
#                     traj_2d,
#                     framenum,
#                     filename,
#                     camera_heading,
#                     ego_heading,
#                     ego_translation,
#                     camera_config,
#                 )
#             )
#     return result
#
# def trajectory_to_timestamp(trajectory):
#     return [traj[0].datetimes for traj in trajectory]
