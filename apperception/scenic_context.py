import ast
import inspect
import os
from typing import Callable
import uncompyle6
import psycopg2
from video_util import *
import datetime 


# Camera node
class ScenicCamera:
    def __init__(self, cam_id, video_file, metadata_id):
        self.cam_id = cam_id 
        self.video_file = video_file
        self.metadata_id = metadata_id
        self.properties = {}

        # Contain objects that still have yet to be added to the backend
        # If user calls recognize, those items will have already been 
        # stored in the backend. These are reserved for objects that users 
        # have not added to the camera.
        self.items = [] 
        self.object_recognition = None


    def add_item(self, item):
        # Add item
        self.items.append(item)

    def add_property(self, properties, property_type, new_prop):
        # Add property
        self.properties[property_type].append(new_prop)

    # Add a default add_recog_obj = True
    def recognize(self, scenic_data_dir):
        # Create object recognition node
        object_rec_node = ScenicObjectRecognition(scenic_data_dir)
        self.object_recognition = object_rec_node
        return object_rec_node
    
# Object Recognition node
class ScenicObjectRecognition:
    def __init__(self, scenic_data_dir):
        self.scenic_data_dir = scenic_data_dir
        self.properties = {}

    def add_properties(self, properties):
        self.properties = properties

class ScenicVideoContext:
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
    def scenic_camera(self, cam_id, video_file, metadata_id):
        camera_node = self.__get_camera(cam_id)
        if not camera_node:
            camera_node = ScenicCamera(cam_id, video_file, metadata_id)
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

    

