from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import psycopg2


@dataclass
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

    def add_item(self, item: Item):
        # Add item
        self.items.append(item)

    def add_property(self, properties, property_type: str, new_prop):
        # TODO: add type annotation
        # Add property
        self.properties[property_type].append(new_prop)

    # Add a default add_recog_obj = True
    def recognize(self, sample_data, annotation):
        # Create object recognition node
        object_rec_node = ObjectRecognition(sample_data, annotation)
        self.object_recognition = object_rec_node
        return object_rec_node


@dataclass
class Item:
    """Item node"""

    item_id: str
    item_type: str
    location: Any  # TODO: what is the type of location?
    properties: dict = field(default_factory=dict)  # TODO: what is the type of properties?


# Object Recognition node
class ObjectRecognition:
    def __init__(self, sample_data, annotation):
        self.sample_data = sample_data
        self.annotation = annotation
        self.properties = {}

    def add_properties(self, properties):
        self.properties = properties


class VideoContext:
    def __init__(self, name: str, units):
        self.root: VideoContext = self
        self.name: str = name
        self.units = units
        self.camera_nodes: Dict[str, Camera] = {}
        self.start_time: datetime.datetime = datetime.datetime(2021, 6, 8, 7, 10, 28)
        self.conn: Optional[psycopg2.connection] = None

    def connect_db(self, host="localhost", user=None, password=None, port=5432, database_name=None):
        """Connect to the database"""
        self.conn = psycopg2.connect(
            database=database_name, user=user, password=password, host=host, port=port
        )

    def get_name(self):
        return self.name

    def get_units(self):
        return self.units

    # Establish camera
    def camera(self, scenic_scene_name):
        camera_node = self.__get_camera(scenic_scene_name)
        if not camera_node:
            camera_node = Camera(scenic_scene_name)
            self.__add_camera(scenic_scene_name, camera_node)
        return camera_node

    def properties(self, cam_id: str, properties, property_type):
        camera_node = self.__get_camera(cam_id)
        if not camera_node:
            return None

        camera_node.add_properties(properties, property_type)
        # Display error

    def get_camera(self, cam_id: str):
        return self.__get_camera(cam_id)

    def __get_camera(self, cam_id: str):
        """Get camera"""
        if cam_id in self.camera_nodes.keys():
            return self.camera_nodes[cam_id]
        return None

    def __add_camera(self, cam_id: str, camera_node: Camera):
        """Add camera"""
        self.camera_nodes[cam_id] = camera_node

    def remove_camera(self, cam_id: str):
        """Remove camera"""
        del self.camera_nodes[cam_id]

    def clear(self):
        """Clear"""
        self.camera_nodes = {}
