from __future__ import annotations
from dataclasses import dataclass, field
import datetime
from typing import Any, Dict, List, Optional

import psycopg2
from tracker import Tracker
from lens import Lens
from point import Point
from video_util import Units


class Camera:
    """Camera node"""

    def __init__(
        self, cam_id: str, point: Point, ratio: float, video_file: str, metadata_id: str, lens: Lens
    ):
        self.cam_id: str = cam_id
        self.ratio: float = ratio
        self.video_file: str = video_file
        self.metadata_id: str = metadata_id
        self.properties: Dict[str, list] = {}  # TODO: what is the type of properties?

        # Contain objects that still have yet to be added to the backend
        # If user calls recognize, those items will have already been
        # stored in the backend. These are reserved for objects that users
        # have not added to the camera.
        self.items: List[Item] = []
        self.object_recognition: Optional[ObjectRecognition] = None
        self.point: Point = point
        self.lens: Lens = lens

    def add_item(self, item: Item):
        # Add item
        self.items.append(item)

    def add_property(self, properties, property_type: str, new_prop):
        # TODO: add type annotation
        # Add property
        self.properties[property_type].append(new_prop)

    def add_lens(self, lens: Lens):
        # Add lens
        self.lens = lens

    def recognize(
        self, algo: str = "Yolo", tracker_type: str = "multi", tracker: Optional[Tracker] = None
    ):
        # Add a default add_recog_obj = True (TODO?)
        # Create object recognition node
        object_rec_node = ObjectRecognition(algo, tracker_type, tracker=None)
        self.object_recognition = object_rec_node
        return object_rec_node


@dataclass
class Item:
    """Item node"""
    item_id: str
    item_type: str
    location: Any  # TODO: what is the type of location?
    properties: dict = field(default_factory=dict)  # TODO: what is the type of properties?


@dataclass
class ObjectRecognition:
    """Object Recognition node"""
    algo: str
    tracker_type: str
    tracker: Optional[Tracker] = None
    bboxes: list = field(default_factory=list)  # TODO: what is the type of bboxes?
    labels: Any = None  # TODO: what is the type of labels?
    tracked_cnt: Any = None  # TODO: what is the type of trackd_cnt?
    properties: Any = None  # TODO: what is the type of properties?

    def add_properties(self, properties):
        self.properties = properties


class VideoContext:
    def __init__(self, name: str, units):
        self.root: VideoContext = self
        self.name: str = name
        self.units: Units = units
        self.camera_nodes: Dict[str, Camera] = {}
        self.start_time: datetime.datetime = datetime.datetime(2021, 6, 8, 7, 10, 28)
        self.conn: Any = None

    def connect_db(self, host="localhost", user=None, password=None, port=5432, database_name=None):
        """Connect to the database"""
        self.conn = psycopg2.connect(
            database=database_name, user=user, password=password, host=host, port=port
        )

    def get_name(self):
        return self.name

    def get_units(self):
        return self.units

    def camera(self, cam_id: str, point: Point, ratio: float, video_file: str, metadata_id: str, lens: Lens):
        """Establish camera"""
        camera_node = self.__get_camera(cam_id)
        if not camera_node:
            camera_node = Camera(cam_id, point, ratio, video_file, metadata_id, lens)
            self.__add_camera(cam_id, camera_node)
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
