import ast
import inspect
import os
from typing import Callable
import uncompyle6
import psycopg2
from video_util import *
import datetime 


# Camera node
class Camera:
    def __init__(self, cam_id, point, ratio, video_file, metadata_id, lens):
        self.cam_id = cam_id 
        self.ratio = ratio
        self.video_file = video_file
        self.metadata_id = metadata_id
        self.properties = {}

        # Contain objects that still have yet to be added to the backend
        # If user calls recognize, those items will have already been 
        # stored in the backend. These are reserved for objects that users 
        # have not added to the camera.
        self.items = [] 
        self.object_recognition = None
        self.point = point
        self.lens = lens

    def add_item(self, item):
        # Add item
        self.items.append(item)

    def add_property(self, properties, property_type, new_prop):
        # Add property
        self.properties[property_type].append(new_prop)

    def add_lens(self, lens):
        # Add lens
        self.lens = lens

    # Add a default add_recog_obj = True
    def recognize(self, algo = 'Yolo', tracker_type = 'multi', tracker = None):
        # Create object recognition node
        object_rec_node = ObjectRecognition(algo, tracker_type, tracker=None)
        self.object_recognition = object_rec_node
        return object_rec_node

# Item node
class Item:
    def __init__(self, item_id, item_type, location):
        self.item_id = item_id
        self.item_type = item_type
        self.location = location
        self.properties = {}


# Object Recognition node
class ObjectRecognition:
    def __init__(self, algo, tracker_type, tracker = None):
        self.algo = algo
        self.tracker_type = tracker_type
        self.tracker = tracker
        # bounding boxes from object recognition
        self.bboxes = []
        self.labels = None
        self.tracked_cnt = None

    def add_properties(self, properties):
        self.properties = properties

class VideoContext:
    def __init__(self, name, units):
        self.root = self
        self.name = name
        self.units = units
        self.camera_nodes = {}
        self.start_time = datetime.datetime(2021, 6, 8, 7, 10, 28)

    # Connect to the database
    def connect_db(self, host='localhost', 
                        user=None,
                        password=None,
                        port=5432,
                        database_name=None):
        self.conn = psycopg2.connect(database=database_name, user=user, 
            password=password, host=host, port=port)
        
    def get_name(self):
        return self.name
    
    def get_units(self):
        return self.units

    # Establish camera
    def camera(self, cam_id, point, ratio, video_file, metadata_id, lens):
        camera_node = self.__get_camera(cam_id)
        if not camera_node:
            camera_node = Camera(cam_id, point, ratio, video_file, metadata_id, lens)
            self.__add_camera(cam_id, camera_node)
        return camera_node

    def properties(self, cam_id, properties, property_type):
        camera_node = self.__get_camera(cam_id)
        if not camera_node:
            return None
     
        camera_node.add_properties(properties, property_type)
       # Display error 

    def get_camera(self, cam_id):
        return self.__get_camera(cam_id)
    
    # Get camera
    def __get_camera(self, cam_id):
        if cam_id in self.camera_nodes.keys():
          return self.camera_nodes[cam_id]
        return None

    # Add camera
    def __add_camera(self, cam_id, camera_node):
        self.camera_nodes[cam_id] = camera_node

    # Remove camera
    def remove_camera(self, cam_id):
        camera_node = self.__get_camera(cam_id)
        self.camera_nodes.remove(camera_node)

    # Clear
    def clear(self):
        self.camera_nodes = []

    

