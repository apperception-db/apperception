import uuid
from enum import Enum
from typing import Set
import datetime

from bounding_box import BoundingBox
from new_db import Database
from new_util import create_camera
from video_context import Camera


class Type(Enum):
    # query type:
    # for example, if we call get_cam(), and we execute the commands from root.
    # when we encounter recognize(), we should not execute it because the inserted object must not be in the final result.
    # we use enum type to determine whether we should execute this node
    CAM, BBOX, TRAJ = 0, 1, 2


class World:
    # all worlds share a db instance
    db = Database()

    def __init__(self):
        self.fn = None
        self.args = None
        self.kwargs = None
        self.parent = None
        self.done = False  # update node
        self.world_id = str(uuid.uuid4())
        self.type: Set[Type] = None

    def get_id(self):
        # to get world id
        return self.world_id

    def recognize(self, camera_node: Camera, recognition_area: BoundingBox):
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

    def filter_traj_type(self, object_type: str):
        new_node = self._create_new_world_and_link()
        new_node.fn = self.db.filter_traj_type
        new_node.type = set([Type.TRAJ])
        new_node.args, new_node.kwargs = [], {"object_type": object_type}
        return new_node

    def interval(self, start, end):
        new_node = self._create_new_world_and_link()

        starttime = str(self.db.start_time + datetime.timedelta(seconds=start))
        endtime = str(self.db.start_time + datetime.timedelta(seconds=end))

        new_node.fn = self.db.interval
        new_node.type = set([Type.BBOX])
        new_node.args, new_node.kwargs = [], {"start": starttime, "end": endtime}
        return new_node

    def add_camera(self, camera_node: Camera):
        """
        1. For update method, we create two nodes: the first node will write to the db, and the second node will retrieve from the db
        2. For the write node, never double write. (so we use done flag)
        ... -> [write] -> [retrive] -> ...
        """
        node1 = self._insert_camera(camera_node=camera_node)
        node2 = node1._retrieve_camera(world_id=node1.world_id)
        return node2

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


if __name__ == "__main__":

    """
    w1 ------ w2 ------------w3----------------------w4-----------------------w5
         cam2(fov=60)  predicate(fov<30)       cam4(fov=120)         predicate(fov<130)
    """
    # w1 = World()

    # c2 = create_camera(cam_id="cam2", fov=60)

    # w2 = w1.add_camera(camera_node=c2)

    # w3 = w2.predicate(condition="query.fov < 30")

    # c4 = create_camera(cam_id="cam4", fov=120)

    # w4 = w3.add_camera(camera_node=c4)

    # w5 = w4.predicate(condition="query.fov < 130")

    # res = w5.get_camera()

    # print(res)

    """
    w1 ------ w2 ----------------- w3 ---------------- w4 --------------------
          recognize              recognize          filter_traj_type         |
              |                                                              |
              -------- get_traj                                              -------- get_traj
    """
    # w1 = World()
    # c2 = create_camera(cam_id="cam2", fov=60)
    # c3 = create_camera(cam_id="cam3", fov=60)
    # # w2 = w1.add_camera(camera_node=c2)
    # w2 = w1.recognize(camera_node=c2, recognition_area=BoundingBox(0, 50, 50, 100))
    # w3 = w2.recognize(camera_node=c3, recognition_area=BoundingBox(0, 50, 50, 100))
    # res2 = w2.get_traj()
    # print(res2)
    # w4 = w3.filter_traj_type("person")
    # res4 = w4.get_traj()
    # print(res4)

    """
    w1 ------ w2 --------------- get_bbox
          recognize
    """
    w1 = World()
    c2 = create_camera(cam_id="cam2", fov=60)
    w2 = w1.recognize(camera_node=c2, recognition_area=BoundingBox(0, 50, 50, 100))

    w3 = w2.interval(0, 3)
    res2 = w3.get_bbox()
    print(res2)

    """
    w1 ------ w2 ------------w3----------------------w4-----------------------w5
         cam2(fov=60)  predicate(fov<30)       cam4(fov=120)         predicate(fov<130)
    """
    # w1 = World()

    # c2 = create_camera(cam_id="cam2", fov=60)

    # w2 = w1.add_camera(camera_node=c2)

    # w3 = w2.predicate(condition="query.fov < 30")

    # c4 = create_camera(cam_id="cam4", fov=120)

    # w4 = w3.add_camera(camera_node=c4)

    # w5 = w4.predicate(condition="query.fov < 130")

    # res = w5.get_len()

    # print(res)
    # print(w4.get_id())
    # print(w3.get_id())
