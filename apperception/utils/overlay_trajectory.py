from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple, Union

if TYPE_CHECKING:
    pass

import cv2
import numpy as np

from apperception.utils import transformation

# from apperception.world import World

FRAME_RATE = 10
FONT = cv2.FONT_HERSHEY_SIMPLEX

"""
    overlay_trajectory()
Outputs videos for all recognized objects.
- is_overlay_headings = If true, overlay the camera, ego, and road headings as part of the video
- is_overlay_road = If true, overlay a marking of the road that the camera is currently on
- is_overlay_objects = If true, overlay a marking on all the filtered objects in the video
- is_keep_whole_video = If true, will output the entire video for any camera that satisfies the
                    condition at any point in time. If false, a video will be outputed for
                    each camera that includes only the frames that meet the filtering condition.
"""


def overlay_trajectory(
    world,
    database,
    images_data_path: Optional[str] = None,
    num_joined_tables: int = 1,
    is_overlay_headings: bool = False,
    is_overlay_road: bool = False,
    is_overlay_objects: bool = False,
    is_keep_whole_video: bool = False,
):
    id_time_camId_filename = world.get_id_time_camId_filename(num_joined_tables=num_joined_tables)
    frames: Dict[Tuple[str, str], List[Any]] = {}
    itemIds = set()
    filenames = set()
    camIds: Dict[str, Dict[str, str]] = {}
    for frame in id_time_camId_filename:
        objId, time, camId, filename = frame
        itemIds.add(objId)
        if camId not in camIds:
            camIds[camId] = {}
        camIds[camId][filename] = time
        filenames.add(filename)
        file_prefix = "-".join(filename.split("/")[:-1])
        if (file_prefix, camId) not in frames:
            frames[(file_prefix, camId)] = []
        frames[(file_prefix, camId)].append(frame)

    if is_keep_whole_video:
        overlay_trajectory_keep_whole(
            world=world,
            database=database,
            camIds=camIds,
            itemIds=itemIds,
            images_data_path=images_data_path,
            is_overlay_headings=is_overlay_headings,
            is_overlay_road=is_overlay_road,
            is_overlay_objects=is_overlay_objects,
        )
        return

    for file_prefix, camId in frames:
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

            camera_config = fetch_camera_config(cam_filename, database)
            camera_config["time"] = time
            if is_overlay_objects:
                frame_im = overlay_objects(frame_im, itemIds, camera_config, database)
            if is_overlay_headings:
                frame_im = overlay_stats(frame_im, camera_config, database)
            if is_overlay_road:
                frame_im = overlay_road(frame_im, camera_config, database)

            if frame_im is not None:
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


def overlay_trajectory_keep_whole(
    world,
    database,
    camIds: Dict[str, Dict[str, str]],
    itemIds: Set[str],
    images_data_path: Optional[str] = None,
    is_overlay_headings: bool = False,
    is_overlay_road: bool = False,
    is_overlay_objects: bool = False,
):
    for camId in camIds:
        filenames = fetch_camera_video(camId=camId, database=database)
        frame_width = None
        frame_height = None
        vid_writer = None
        for cam_filename in filenames:
            filename_no_prefix = cam_filename.split("/")[-1]
            prefix = "-".join(cam_filename.split("/")[:-1])
            if images_data_path is not None:
                filename = images_data_path + "/" + filename_no_prefix
            else:
                filename = cam_filename
            frame_im = cv2.imread(filename)
            camera_config = fetch_camera_config(cam_filename, database)
            if is_overlay_objects and cam_filename in camIds[camId]:
                camera_config["time"] = camIds[camId][cam_filename]
                frame_im = overlay_objects(frame_im, itemIds, camera_config, database)
            if is_overlay_headings:
                frame_im = overlay_stats(frame_im, camera_config, database)
            if is_overlay_road:
                frame_im = overlay_road(frame_im, camera_config, database)

            if frame_im is not None:
                if vid_writer is None:
                    frame_height, frame_width = frame_im.shape[:2]
                    vid_writer = cv2.VideoWriter(
                        "./output/" + prefix + "." + camId + "-whole" + ".mp4",
                        cv2.VideoWriter_fourcc("m", "p", "4", "v"),
                        FRAME_RATE,
                        (frame_width, frame_height),
                    )
                vid_writer.write(frame_im)
        if vid_writer is not None:
            vid_writer.release()
    # for itemId in itemCameras:
    #     cameraIds = itemCameras[cameraIds]
    #     for cameraId in cameraIds:


##### SQL Utils #####


def fetch_camera_video(camId: str, database):
    query = f"""
    SELECT
        fileName
    FROM Cameras
    WHERE
        cameraId = '{camId}'
    ORDER BY frameNum ASC;
    """
    result = database.execute(query)
    filenames = [x[0] for x in result]
    return filenames


def fetch_camera_config(filename: str, database):
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
        cameraRotation,
        cameraIntrinsic,
        frameNum,
        fileName,
        cameraHeading,
        egoHeading,
        timestamp,
        roadDirection
    FROM Cameras
    WHERE
        fileName = '{filename}'
    ORDER BY cameraId ASC, frameNum ASC;
    """
    result = database.execute(query)[0]
    print(result)
    camera_config = {
        "cameraId": result[0],
        "egoTranslation": result[1],
        "egoRotation": result[2],
        "cameraTranslation": result[3],
        "cameraRotation": result[4],
        "cameraIntrinsic": result[5],
        "frameNum": result[6],
        "fileName": result[7],
        "cameraHeading": result[8],
        "egoHeading": result[9],
        "timestamp": result[10],
        "roadDirection": result[11],
    }
    return camera_config


def fetch_camera_trajectory(video_name: str, database):
    query = f"""
    CREATE OR REPLACE FUNCTION ST_XYZ (g geometry) RETURNS real[] AS $$
        BEGIN
            RETURN ARRAY[ST_X(g), ST_Y(g), ST_Z(g)];
        END;
    $$ LANGUAGE plpgsql;

    SELECT
        cameraId,
        ST_XYZ(egoTranslation),
        frameNum,
        timestamp,
        fileName,
        cameraHeading,
        egoHeading
    FROM Cameras
    WHERE
        fileName LIKE '%{video_name}%'
    ORDER BY cameraId ASC, frameNum ASC;
    """
    result = database._execute_query(query)
    camera_config = []
    for row in result:
        camera_config.append(
            {
                "cameraId": row[0],
                "egoTranslation": row[1],
                "frameNum": row[2],
                "timestamp": row[3],
                "fileName": row[4],
                "cameraHeading": row[5],
                "egoHeading": row[6],
            }
        )
    return camera_config


def fetch_trajectory(itemId: str, time: str, database):
    query = f"""
        CREATE OR REPLACE FUNCTION ST_XYZ (g geometry) RETURNS real[] AS $$
        BEGIN
            RETURN ARRAY[ST_X(g), ST_Y(g), ST_Z(g)];
        END;
        $$ LANGUAGE plpgsql;

        SELECT ST_XYZ(valueAtTimestamp(trajCentroids, '{time}'))
        FROM Item_General_Trajectory as final
        WHERE itemId = '{itemId}';
        """

    traj = database.execute(query)[0][0]
    return traj


##### CV2 Overlay Utils #####
def overlay_objects(frame, itemIds: Set[str], camera_config, database):
    time = camera_config["time"]
    pixels = {}
    for itemId in itemIds:
        current_traj_point = fetch_trajectory(itemId=itemId, time=time, database=database)

        if None not in current_traj_point:
            current_pixel = world_to_pixel(camera_config, current_traj_point)

            pixels[itemId] = current_pixel

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


def overlay_stats(frame, camera_config, database):
    ego_translation = camera_config["egoTranslation"]
    ego_heading = camera_config["egoHeading"]
    camera_heading = camera_config["cameraHeading"]
    cam_road_dir = database.road_direction(ego_translation[0], ego_translation[1], ego_heading)[0][
        0
    ]

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


def overlay_road(frame, camera_config, database):
    ego_translation = camera_config["egoTranslation"]
    camera_road_coords = database.road_coords(ego_translation[0], ego_translation[1])[0][0]
    pixel_start_enc = world_to_pixel(
        camera_config, (camera_road_coords[0], camera_road_coords[1], 0)
    )
    pixel_end_enc = world_to_pixel(camera_config, (camera_road_coords[2], camera_road_coords[3], 0))
    pixel_start = tuple([int(pixel_start_enc[0][0]), int(pixel_start_enc[1][0])])
    pixel_end = tuple([int(pixel_end_enc[0][0]), int(pixel_end_enc[1][0])])
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
    traj_2d = transformation.transformation(world_coords, camera_config)
    return traj_2d
