import datetime
import uuid
from enum import Enum
from typing import Any, Dict, Set

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from bounding_box import WHOLE_FRAME, BoundingBox
from lens import Lens
from new_db import Database
from point import Point
from video_context import Camera

matplotlib.use("Qt5Agg")
print("get backend", matplotlib.get_backend())
# plt.figure()
# plt.plot([1,2,3,4])
# plt.show()


class Type(Enum):
    # query type: for example, if we call get_cam(), and we execute the commands from root. when we encounter
    # recognize(), we should not execute it because the inserted object must not be in the final result. we use enum
    # type to determine whether we should execute this node
    CAM, BBOX, TRAJ = 0, 1, 2


BASE_VOLUME_QUERY_TEXT = "STBOX Z(({x1}, {y1}, {z1}),({x2}, {y2}, {z2}))"


class World:
    # all worlds share a db instance
    db = Database()
    camera_nodes: Dict[str, Camera] = {}

    def __init__(self):
        self.fn = None
        self.args = None
        self.kwargs = None
        self.parent = None
        self.done = False  # update node
        self.world_id = str(uuid.uuid4())
        self.type: Set[Type] = None

    def overlay_trajectory(self, cam_id, trajectory):
        matplotlib.use(
            "Qt5Agg"
        )  # FIXME: matplotlib backend is agg here (should be qt5agg). Why is it overwritten?
        print("get backend", matplotlib.get_backend())
        camera = World.camera_nodes[cam_id]
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

    def select_intersection_of_interest_or_use_default(self, cam_id, default=True):
        camera = self.camera_nodes[cam_id]
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

    def get_id(self):
        # to get world id
        return self.world_id

    def recognize(self, cam_id: str, recognition_area: BoundingBox = WHOLE_FRAME):
        assert cam_id in World.camera_nodes

        camera_node = World.camera_nodes[cam_id]
        node1 = self._insert_bbox_traj(camera_node=camera_node, recognition_area=recognition_area)
        node2 = node1._retrieve_bbox(world_id=node1.world_id)
        node3 = node2._retrieve_traj(world_id=node1.world_id)
        return node3

    def _insert_bbox_traj(self, camera_node: Camera, recognition_area: BoundingBox):
        new_node = self._create_new_world_and_link()
        new_node.fn = self.db.insert_bbox_traj
        new_node.type = set([Type.TRAJ, Type.BBOX])
        new_node.args, new_node.kwargs = [], {
            "world_id": new_node.world_id,
            "camera_node": camera_node,
            "recognition_area": recognition_area,
        }
        return new_node

    def _retrieve_bbox(self, world_id: str):
        new_node = self._create_new_world_and_link()
        new_node.fn = self.db.retrieve_bbox
        new_node.type = set([Type.BBOX])
        new_node.args, new_node.kwargs = [], {"world_id": world_id}
        return new_node

    def _retrieve_traj(self, world_id: str):
        new_node = self._create_new_world_and_link()
        new_node.fn = self.db.retrieve_traj
        new_node.type = set([Type.TRAJ])
        new_node.args, new_node.kwargs = [], {"world_id": world_id}
        return new_node

    def get_bbox(self):
        new_node = self._create_new_world_and_link()
        new_node.fn = self.db.get_bbox
        new_node.type = set([Type.BBOX])
        new_node.args, new_node.kwargs = [], {}
        return new_node._execute_from_root(Type.BBOX)

    def get_traj(self):
        new_node = self._create_new_world_and_link()
        new_node.fn = self.db.get_traj
        new_node.type = set([Type.TRAJ])
        new_node.args, new_node.kwargs = [], {}
        return new_node._execute_from_root(Type.TRAJ)

    def get_traj_key(self):
        new_node = self._create_new_world_and_link()
        new_node.fn = self.db.get_traj_key
        new_node.type = set([Type.TRAJ])
        new_node.args, new_node.kwargs = [], {}
        return new_node._execute_from_root(Type.TRAJ)

    def filter_traj_type(self, object_type: str):
        new_node = self._create_new_world_and_link()
        new_node.fn = self.db.filter_traj_type
        new_node.type = set([Type.TRAJ])
        new_node.args, new_node.kwargs = [], {"object_type": object_type}
        return new_node

    def filter_traj_volume(self, volume: str):
        new_node = self._create_new_world_and_link()
        new_node.fn = self.db.filter_traj_volume
        new_node.type = set([Type.TRAJ])
        new_node.args, new_node.kwargs = [], {"volume": volume}
        return new_node

    def add_camera(
        self,
        cam_id: str,
        location: Point,
        ratio: float,
        video_file: str,
        metadata_identifier: str,
        lens: Lens,
    ):
        """
        1. For update method, we create two nodes: the first node will write to the db, and the second node will retrieve from the db
        2. For the write node, never double write. (so we use done flag)
        ... -> [write] -> [retrive] -> ...
        """
        camera_node = Camera(cam_id, location, ratio, video_file, metadata_identifier, lens)
        World.camera_nodes[cam_id] = camera_node

        node1 = self._insert_camera(camera_node=camera_node)
        node2 = node1._retrieve_camera(world_id=node1.world_id)
        return node2

    def interval(self, start, end):
        new_node = self._create_new_world_and_link()

        starttime = str(self.db.start_time + datetime.timedelta(seconds=start))
        endtime = str(self.db.start_time + datetime.timedelta(seconds=end))

        new_node.fn = self.db.interval
        new_node.type = set([Type.BBOX])
        new_node.args, new_node.kwargs = [], {"start": starttime, "end": endtime}
        return new_node

    def add_properties(self, cam_id: str, properties: Any):
        self.camera_nodes[cam_id].add_property(properties)

    def predicate(self, condition: str):
        new_node = self._create_new_world_and_link()
        new_node.fn = self.db.filter_cam
        new_node.type = set([Type.CAM])
        new_node.args, new_node.kwargs = [], {"condition": condition}
        return new_node

    def _insert_camera(self, camera_node: Camera):
        new_node = self._create_new_world_and_link()
        new_node.fn = self.db.insert_cam
        new_node.type = set([Type.CAM])
        new_node.args, new_node.kwargs = [], {
            "camera_node": camera_node,
            "world_id": new_node.world_id,
        }
        return new_node

    def _retrieve_camera(self, world_id: str):
        new_node = self._create_new_world_and_link()
        new_node.fn = self.db.retrieve_cam
        new_node.type = set([Type.CAM])
        new_node.args, new_node.kwargs = [], {"world_id": world_id}
        return new_node

    def get_len(self):
        new_node = self._create_new_world_and_link()
        new_node.fn = self.db.get_len
        new_node.type = set([Type.CAM])
        new_node.args, new_node.kwargs = [], {}
        return new_node._execute_from_root(Type.CAM)

    def get_camera(self):
        new_node = self._create_new_world_and_link()
        new_node.fn = self.db.get_cam
        new_node.type = set([Type.CAM])
        new_node.args, new_node.kwargs = [], {}
        return new_node._execute_from_root(Type.CAM)

    def _create_new_world_and_link(self):
        new_world = World()
        new_world.parent = self
        return new_world

    def _execute_from_root(self, type: Type):
        nodes = []
        curr = self
        res = None
        query = ""

        # collect all the nodes til the root
        while curr:
            nodes.append(curr)
            curr = curr.parent

        # execute the nodes from the root
        for node in nodes[::-1]:
            # root
            if node.fn is None:
                continue
            # if different type => pass
            if type not in node.type:
                continue
            # treat update method differently
            elif node.fn == self.db.insert_cam or node.fn == self.db.insert_bbox_traj:
                if not node.done:
                    node._execute()
                    node.done = True
            else:
                # print(query)
                query = node._execute(query=query)

        res = query
        return res

    def _execute(self, *args, **kwargs):
        # print("executing fn = {}, with args = {} and kwargs = {}".format(self.fn, self.args, self.kwargs))
        return self.fn(*self.args, *args, **self.kwargs, **kwargs)

    def _print_til_root(self):
        curr = self
        while curr:
            # print(curr)
            curr = curr.parent

    def __str__(self):
        return "fn={}\nargs={}\nkwargs={}\ndone={}\nworld_id={}\n".format(
            self.fn, self.args, self.kwargs, self.done, self.world_id
        )
