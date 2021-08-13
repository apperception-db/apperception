from typing import Callable
import uncompyle6
import psycopg2
from metadata_context import *
from video_context import *
import copy
from world_executor import WorldExecutor

BASE_VOLUME_QUERY_TEXT = "stbox \'STBOX Z(({x1}, {y1}, {z1}),({x2}, {y2}, {z2}))\'"

class World:

    def __init__(self, name, units):
        self.VideoContext = VideoContext(name, units)
        self.MetadataContext = MetadataContext(single_mode=False)
        self.MetadataContext.start_time = self.VideoContext.start_time
        self.GetVideo = False
        # self.AccessedVideoContext = False
    
    def fetch_tasm(self):
        return self.VideoContext.tasm
    
    def get_camera(self, cam_id=[]):
        world_executor = WorldExecutor(self)
        # Change depending if you're on docker or not 
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
 
    def item(self, item_id, cam_id, item_type, location):
        new_context = copy.deepcopy(self)
        new_context.VideoContext.item(item_id, cam_id, item_type, location)
        return new_context

    def camera(self, cam_id, location, ratio, video_file, metadata_identifier, lens):
        new_context = copy.deepcopy(self)
        new_context.VideoContext.camera(cam_id, location, ratio, video_file, metadata_identifier, lens)
        return new_context

    def add_properties(self, cam_id, properties, property_type):
        new_context = copy.deepcopy(self)
        new_context.VideoContext.properties(cam_id, properties, property_type)
        return new_context

    def recognize(self, cam_id, algo ='Yolo', tracker_type = 'multi', tracker = None):
        new_context = copy.deepcopy(self)
        new_context.VideoContext.camera_nodes[cam_id].recognize(algo, tracker_type, tracker)
        return new_context

#########################
### Metadata Context ####
#########################

    def get_columns(self, *argv, distinct = False):
        new_context = copy.deepcopy(self)
        new_context.MetadataContext.get_columns(argv, distinct)
        return new_context

    def predicate(self, p, evaluated_var = {}):
        new_context = copy.deepcopy(self)
        new_context.MetadataContext.predicate(p, evaluated_var)
        return new_context

    def selectkey(self, distinct = False):
        new_context = copy.deepcopy(self)
        new_context.MetadataContext.selectkey(distinct)
        return new_context

    def get_trajectory(self, interval = [], distinct = False):
        new_context = copy.deepcopy(self)
        new_context.MetadataContext.get_trajectory(interval, distinct)
        return new_context

    def get_geo(self, interval = [], distinct = False):
        new_context = copy.deepcopy(self)
        new_context.MetadataContext.get_geo(interval, distinct)
        return new_context
        
    def get_time(self, distinct = False):
        new_context = copy.deepcopy(self)
        new_context.MetadataContext.get_time(distinct)
        return new_context  
    
    def get_distance(self, interval = [], distinct = False):
        new_context = copy.deepcopy(self)
        new_context.MetadataContext.distance(interval, distinct)
        return new_context
        
    def get_speed(self, interval = [], distinct = False):
        new_context = copy.deepcopy(self)
        new_context.MetadataContext.get_speed(interval, distinct)
        return new_context
    
    def get_video(self, cam_id=[]):
        # Go through all the cameras in 'filtered' world and obtain videos 
        new_context = copy.deepcopy(self)
        new_context.GetVideo = True
        ## get camera gives the direct results from the data base
        new_context.get_video_cams = self.get_camera(cam_id)
        return new_context

    def interval(self, time_interval):
        new_context = copy.deepcopy(self)
        new_context.MetadataContext.interval(time_interval)
        return new_context
    
    def execute(self):
        world_executor = WorldExecutor(self)
        # Change depending if you're on docker or not 
        world_executor.connect_db(user="docker", password="docker", database_name="mobilitydb")
        return world_executor.execute()

    def select_intersection_of_interest_or_use_default(self, cam_id, default=True):
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
            cv2.resizeWindow('Frame', 384, 216)
            initBB = cv2.selectROI("Frame", frame, fromCenter=False)
            print(initBB)
            cv2.destroyAllWindows()
            print("world coordinate #1")
            tl = camera.lens.pixel_to_world(initBB[0], 1)
            print(tl)
            x1, y1, z1 = tl
            print("world coordinate #2")
            br = camera.lens.pixel_to_world((initBB[0][0]+initBB[1][0], initBB[0][1]+initBB[1][1]), 1)
            print(br)
            x2, y2, z2 = br
        return BASE_VOLUME_QUERY_TEXT.format(x1=x1, y1=y1, z1=z1, x2=x2, y2=y2, z2=z2)
  