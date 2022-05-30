import copy
import datetime

import cv2
import matplotlib
import numpy as np
from scenic_util import transformation

from apperception.legacy.metadata_context import MetadataContext
from apperception.legacy.video_context import VideoContext
from apperception.legacy.world_executor import WorldExecutor

matplotlib.use("Qt5Agg")

BASE_VOLUME_QUERY_TEXT = "stbox 'STBOX Z(({x1}, {y1}, {z1}),({x2}, {y2}, {z2}))'"
world_executor = WorldExecutor()


class World:
    def __init__(self, name, units, enable_tasm=False):
        self.VideoContext = VideoContext(name, units)
        self.MetadataContext = MetadataContext(single_mode=False)
        self.MetadataContext.start_time = self.VideoContext.start_time
        self.GetVideo = False
        self.enable_tasm = enable_tasm
        # self.AccessedVideoContext = False

    def get_camera(self, scene_name, frame_num):
        # Change depending if you're on docker or not
        # TODO: fix get_camera in scenic_world_executor.py
        if self.enable_tasm:
            world_executor.connect_db(
                port=5432, user="docker", password="docker", database_name="mobilitydb"
            )
        else:
            world_executor.connect_db(user="docker", password="docker", database_name="mobilitydb")
        return world_executor.get_camera(scene_name, frame_num)

    #########################
    ###   Video Context  ####
    #########################
    def get_lens(self, cam_id=""):
        return self.get_camera(cam_id).lens

    def get_name(self):
        return self.VideoContext.get_name()

    def get_units(self):
        return self.VideoContext.get_units()

    # TODO: should be add_item / add_camera?
    def item(self, item_id, cam_id, item_type, location):
        new_context = copy.deepcopy(self)
        new_context.VideoContext.item(item_id, cam_id, item_type, location)
        return new_context

    def camera(self, scenic_scene_name: str):
        new_context = copy.deepcopy(self)
        new_context.VideoContext.camera(scenic_scene_name)
        return new_context

    def add_properties(self, cam_id, properties, property_type):
        new_context = copy.deepcopy(self)
        new_context.VideoContext.properties(cam_id, properties, property_type)
        return new_context

    def recognize(self, cam_id, sample_data, annotation):
        new_context = copy.deepcopy(self)
        new_context.VideoContext.camera_nodes[cam_id].recognize(sample_data, annotation)
        return new_context

    #########################
    ### Metadata Context ####
    #########################

    def get_columns(self, *argv, distinct=False):
        new_context = copy.deepcopy(self)
        new_context.MetadataContext.get_columns(argv, distinct)
        return new_context

    def predicate(self, p, evaluated_var={}):
        new_context = copy.deepcopy(self)
        new_context.MetadataContext.predicate(p, evaluated_var)
        return new_context

    def selectkey(self, distinct=False):
        new_context = copy.deepcopy(self)
        new_context.MetadataContext.selectkey(distinct)
        return new_context

    def get_trajectory(self, interval=[], distinct=False):
        new_context = copy.deepcopy(self)
        new_context.MetadataContext.get_trajectory(interval, distinct)
        return new_context

    def get_geo(self, interval=[], distinct=False):
        new_context = copy.deepcopy(self)
        new_context.MetadataContext.get_geo(interval, distinct)
        return new_context

    def get_time(self, distinct=False):
        new_context = copy.deepcopy(self)
        new_context.MetadataContext.get_time(distinct)
        return new_context

    def get_distance(self, interval=[], distinct=False):
        new_context = copy.deepcopy(self)
        new_context.MetadataContext.distance(interval, distinct)
        return new_context

    def get_speed(self, interval=[], distinct=False):
        new_context = copy.deepcopy(self)
        new_context.MetadataContext.get_speed(interval, distinct)
        return new_context

    def get_video(self, cam_id=[]):
        # Go through all the cameras in 'filtered' world and obtain videos
        new_context = copy.deepcopy(self)
        new_context.GetVideo = True
        # get camera gives the direct results from the data base
        new_context.get_video_cams = self.get_camera(cam_id)
        return new_context

    def interval(self, time_interval):
        new_context = copy.deepcopy(self)
        new_context.MetadataContext.interval(time_interval)
        return new_context

    def execute(self):
        world_executor.create_world(self)
        if self.enable_tasm:
            world_executor.enable_tasm()
            print("successfully enable tasm during execution time")
            # Change depending if you're on docker or not
            world_executor.connect_db(
                port=5432, user="docker", password="docker", database_name="mobilitydb"
            )
        else:
            world_executor.connect_db(user="docker", password="docker", database_name="mobilitydb")
        return world_executor.execute()

    def select_intersection_of_interest_or_use_default(self, cam_id, default=True):
        print(self.VideoContext.camera_nodes)
        camera = self.VideoContext.camera_nodes[cam_id]
        video_file = camera.video_file
        if default:
            x1, y1, z1 = 0.01082532, 2.59647246, 0
            x2, y2, z2 = 3.01034039, 3.35985782, 2
        else:
            vs = cv2.VideoCapture(video_file)
            frame = vs.read()
            frame = frame[1]
            cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Frame", 384, 216)
            initBB = cv2.selectROI("Frame", frame, fromCenter=False)
            print(initBB)
            cv2.destroyAllWindows()
            print("world coordinate #1")
            tl = camera.lens.pixel_to_world(initBB[:2], 1)
            print(tl)
            x1, y1, z1 = tl
            print("world coordinate #2")
            br = camera.lens.pixel_to_world((initBB[0] + initBB[2], initBB[1] + initBB[3]), 1)
            print(br)
            x2, y2, z2 = br
        return BASE_VOLUME_QUERY_TEXT.format(x1=x1, y1=y1, z1=0, x2=x2, y2=y2, z2=2)

    def overlay_trajectory(self, scene_name, trajectory, video_file: str):
        frame_num = self.trajectory_to_frame_num(trajectory)
        # frame_num is int[[]], hence camera_info should also be [[]]
        camera_info = []
        for cur_frame_num in frame_num:
            camera_info.append(
                self.get_camera(scene_name, cur_frame_num)
            )  # TODO: fetch_camera_info in scenic_utils.py
        assert len(camera_info) == len(frame_num)
        assert len(camera_info[0]) == len(frame_num[0])
        # overlay_info = self.get_overlay_info(trajectory, camera_info)
        # TODO: fix the following to overlay the 2d point onto the frame
        # TODO: clean up: this for loop does not work anymore because we are not passing in camera
        # for traj in trajectory:
        #     current_trajectory = np.asarray(traj[0])
        #     frame_points = camera.lens.world_to_pixels(current_trajectory.T).T
        #     vs = cv2.VideoCapture(video_file)
        #     frame = vs.read()
        #     frame = cv2.cvtColor(frame[1], cv2.COLOR_BGR2RGB)
        #     for point in frame_points.tolist():
        #         cv2.circle(frame, tuple([int(point[0]), int(point[1])]), 3, (255, 0, 0))
        #     plt.figure()
        #     plt.imshow(frame)
        #     plt.show()

    def trajectory_to_frame_num(self, trajectory):
        """
        fetch the frame number from the trajectory
        1. get the time stamp field from the trajectory
        2. convert the time stamp to frame number
            Refer to 'convert_datetime_to_frame_num' in 'video_util.py'
        3. return the frame number
        """
        frame_num = []
        start_time = self.MetadataContext.start_time
        for traj in trajectory:
            current_trajectory = traj[0]
            date_times = current_trajectory["datetimes"]
            frame_num.append(
                [
                    (
                        datetime.datetime.strptime(t, "%Y-%m-%dT%H:%M:%S+00").replace(tzinfo=None)
                        - start_time
                    ).total_seconds()
                    for t in date_times
                ]
            )
        return frame_num

    def get_overlay_info(self, trajectory, camera_info):
        """
        overlay each trajectory 3d coordinate on to the frame specified by the camera_info
        1. for each trajectory, get the 3d coordinate
        2. get the camera_info associated to it
        3. implement the transformation function from 3d to 2d
            given the single centroid point and camera configuration
            refer to TODO in "senic_utils.py"
        4. return a list of (2d coordinate, frame name/filename)
        """
        result = []
        for traj_num in range(len(trajectory)):
            traj_obj = trajectory[traj_num][0]  # traj_obj means the trajectory of current object
            traj_obj_3d = traj_obj["coordinates"]  # 3d coordinate list of the object's trajectory
            camera_info_obj = camera_info[
                traj_num
            ]  # camera info list corresponding the 3d coordinate
            traj_obj_2d = []  # 2d coordinate list
            for index in range(len(camera_info_obj)):
                cur_camera_info = camera_info_obj[
                    index
                ]  # camera info of the obejct in one point of the trajectory
                centroid_3d = np.array(traj_obj_3d[index])  # one point of the trajectory in 3d
                # in order to fit into the function transformation, we develop a dictionary called camera_config
                camera_config = {}
                camera_config["egoTranslation"] = cur_camera_info[1]
                camera_config["egoRotation"] = np.array(cur_camera_info[2])
                camera_config["cameraTranslation"] = cur_camera_info[3]
                camera_config["cameraRotation"] = np.array(cur_camera_info[4])
                camera_config["cameraIntrinsic"] = np.array(cur_camera_info[5])
                traj_2d = transformation(
                    centroid_3d, camera_config
                )  # one point of the trajectory in 2d

                framenum = cur_camera_info[6]
                filename = cur_camera_info[7]
                traj_obj_2d.append((traj_2d, framenum, filename))
            result.append(traj_obj_2d)
        return result
