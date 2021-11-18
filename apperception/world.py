import copy

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from bounding_box import WHOLE_FRAME, BoundingBox
from metadata_context import MetadataContext
from tracker import Tracker
from video_context import VideoContext
from world_executor import WorldExecutor

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

    def get_camera(self, cam_id=[]):
        # Change depending if you're on docker or not
        if self.enable_tasm:
            world_executor.connect_db(
                port=5432, user="docker", password="docker", database_name="mobilitydb"
            )
        else:
            world_executor.connect_db(user="docker", password="docker", database_name="mobilitydb")
        return world_executor.get_camera(cam_id)

    #########################
    ###   Video Context  ####
    #########################
    # TODO(@Vanessa): Add a helper function
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

    def camera(self, cam_id, location, ratio, video_file, metadata_identifier, lens):
        new_context = copy.deepcopy(self)
        new_context.VideoContext.camera(
            cam_id, location, ratio, video_file, metadata_identifier, lens
        )
        return new_context

    def add_properties(self, cam_id, properties, property_type):
        new_context = copy.deepcopy(self)
        new_context.VideoContext.properties(cam_id, properties, property_type)
        return new_context

    def recognize(
        self,
        cam_id: str,
        algo: str = "Yolo",
        tracker_type: str = "multi",
        tracker: Tracker = None,
        recognition_area: BoundingBox = WHOLE_FRAME,  # bounding box value in percentage not pixel
    ):
        new_context = copy.deepcopy(self)
        new_context.VideoContext.camera_nodes[cam_id].recognize(
            algo, tracker_type, tracker, recognition_area
        )
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

    def overlay_trajectory(self, cam_id, trajectory):
        camera = self.VideoContext.get_camera(cam_id)
        video_file = camera.video_file
        for traj in trajectory:
            current_trajectory = np.asarray(traj[0])
            frame_points = camera.lens.world_to_pixels(current_trajectory.T).T
            vs = cv2.VideoCapture(video_file)
            frame = vs.read()
            frame = cv2.cvtColor(frame[1], cv2.COLOR_BGR2RGB)
            for point in frame_points.tolist():
                cv2.circle(frame, tuple([int(point[0]), int(point[1])]), 3, (255, 0, 0))
            plt.figure()
            plt.imshow(frame)
            plt.show()
