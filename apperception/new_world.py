from __future__ import annotations

import datetime
import inspect
import uuid
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import cv2
import numpy as np
from data_types import Camera, QueryType, Trajectory
from new_db import Database
from pypika import Table
from pypika.dialects import SnowflakeQuery
from scenic_util import FetchCameraTuple, transformation

# matplotlib.use("Qt5Agg")
# print("get backend", matplotlib.get_backend())


class World:
    # all worlds share a db instance
    db = Database()
    camera_nodes: Dict[str, Camera] = {}

    _parent: Optional[World]
    _name: str
    _fn: Tuple[Optional[Callable]]
    _kwargs: dict[str, Any]
    _done: bool
    _world_id: str
    _timestamp: datetime.datetime
    _types: set[QueryType]
    _materialized: bool

    def __init__(
        self,
        world_id: str,
        timestamp: datetime.datetime,
        name: Optional[str] = None,
        parent: Optional[World] = None,
        fn: Optional[Union[str, Callable]] = None,
        kwargs: Optional[dict[str, Any]] = None,
        done: bool = False,
        types: Optional[Set[QueryType]] = None,
        materialized: bool = False,
    ):
        self._parent = parent
        self._name = "" if name is None else name
        self._fn = (fn if fn is None else (getattr(self.db, fn) if isinstance(fn, str) else fn),)
        self._kwargs = {} if kwargs is None else kwargs
        self._done = done  # update node
        self._world_id = world_id
        self._timestamp = timestamp
        self._types = set() if types is None else types
        self._materialized = materialized

    def overlay_trajectory(self, scene_name: str, trajectory, object_id: str):
        frame_timestamps = trajectory_to_timestamp(trajectory)
        # camera_info is a list of mappings from timestamps to list of (frame_num, cameras)
        camera_info: List[Dict[int, List["FetchCameraTuple"]]] = []
        for index, cur_frame_timestamp in enumerate(frame_timestamps):
            current_cameras = self.db.fetch_camera(scene_name, cur_frame_timestamp)
            camera_info.append({})
            for x in current_cameras:
                if x[6] in camera_info[index]:
                    camera_info[index][x[6]].append(x)
                else:
                    camera_info[index][x[6]] = [x]
        camera_info_2 = [[x[y] for y in sorted(x)] for x in camera_info]

        overlay_info = get_overlay_info(trajectory, camera_info_2)
        # TODO: fix the following to overlay the 2d point onto the frame
        results: List[Dict[str, List[Tuple[np.ndarray, int, str]]]] = []
        frame_width = None
        frame_height = None
        for i, traj in enumerate(overlay_info):
            results.append({})
            for frame_num in traj:
                for frame in frame_num:
                    if frame_width is None:
                        frame_im = cv2.imread(frame[2])
                        frame_height, frame_width = frame_im.shape[:2]
                    file_suffix = frame[2].split("/")[1]
                    if file_suffix in results[i]:
                        results[i][file_suffix].append(frame)
                    else:
                        results[i][file_suffix] = [frame]

        for i in range(len(results)):
            for file_suffix in results[i]:
                vid_writer = cv2.VideoWriter(
                    "./output/" + object_id + "." + file_suffix + ".mp4",
                    cv2.VideoWriter_fourcc("m", "p", "4", "v"),
                    30,
                    (frame_width, frame_height),
                )
                for frame in results[i][file_suffix]:
                    frame_im = cv2.imread(frame[2])
                    cv2.circle(
                        frame_im,
                        tuple([int(frame[0][0][0]), int(frame[0][1][0])]),
                        10,
                        (0, 255, 0),
                        -1,
                    )
                    vid_writer.write(frame_im)
                vid_writer.release()

    def recognize(self, camera: Camera, annotation):
        node1 = self._insert_bbox_traj(camera=camera, annotation=annotation)
        node2 = node1._retrieve_bbox(camera_id=camera.id)
        node3 = node2._retrieve_traj(camera_id=camera.id)
        return node3

    def get_video(self, cam_ids: List[str] = [], boxed: bool = False):
        return derive_world(
            self,
            {QueryType.TRAJ},
            self.db.get_video,
            cams=[World.camera_nodes[cam_id] for cam_id in cam_ids],
            boxed=boxed,
        )._execute_from_root(QueryType.TRAJ)

    def get_bbox(self):
        return derive_world(
            self,
            {QueryType.BBOX},
            self.db.get_bbox,
        )._execute_from_root(QueryType.BBOX)

    def get_traj(self) -> List[List[Trajectory]]:
        return derive_world(
            self,
            {QueryType.TRAJ},
            self.db.get_traj,
        )._execute_from_root(QueryType.TRAJ)

    def get_traj_key(self):
        return derive_world(
            self,
            {QueryType.TRAJ},
            self.db.get_traj_key,
        )._execute_from_root(QueryType.TRAJ)

    def get_headings(self) -> List[List[List[float]]]:
        # TODO: Optimize operations with NumPy if possible
        trajectories = self.get_traj()
        headings: List[List[List[float]]] = []
        for trajectory in trajectories:
            _headings: List[List[float]] = []
            for traj in trajectory:
                __headings: List[float] = []
                for j in range(1, len(traj.coordinates)):
                    prev_pos = traj.coordinates[j - 1]
                    current_pos = traj.coordinates[j]
                    heading = 0.0
                    if current_pos[1] != prev_pos[1]:
                        heading = np.arctan2(
                            current_pos[1] - prev_pos[1], current_pos[0] - prev_pos[0]
                        )
                    # convert to degrees from radian
                    heading *= 180 / np.pi
                    # converting such that all headings are positive
                    heading = (heading + 360) % 360
                    __headings.append(heading)
                _headings.append(__headings)
            headings.append(_headings)
        return headings

    def get_distance(self, start: datetime.datetime, end: datetime.datetime):
        return derive_world(
            self,
            {QueryType.TRAJ},
            self.db.get_distance,
            start=str(start),
            end=str(end),
        )._execute_from_root(QueryType.TRAJ)

    def get_speed(self, start: datetime.datetime, end: datetime.datetime):
        return derive_world(
            self,
            {QueryType.TRAJ},
            self.db.get_speed,
            start=str(start),
            end=str(end),
        )._execute_from_root(QueryType.TRAJ)

    def add_camera(self, camera: Camera):
        """
        1. For update method, we create two nodes: the first node will write to the db, and the second node will retrieve from the db
        2. For the write node, never double write. (so we use done flag)
        ... -> [write] -> [retrive] -> ...
        """
        node1 = self._insert_camera(camera=camera)
        node2 = node1._retrieve_camera(camera_id=camera.id)
        return node2

    def filter(self, predicate: Union[str, Callable]):
        return derive_world(
            self,
            {QueryType.TRAJ, QueryType.BBOX},
            self.db.filter,
            predicate=predicate,
        )

    def exclude(self, world: World):
        return derive_world(self, {QueryType.TRAJ, QueryType.BBOX}, self.db.exclude, world=world)

    def get_len(self):
        return derive_world(
            self,
            {QueryType.CAM},
            self.db.get_len,
        )._execute_from_root(QueryType.CAM)

    def get_camera(self):
        return derive_world(
            self,
            {QueryType.CAM},
            self.db.get_cam,
        )._execute_from_root(QueryType.CAM)

    def get_bbox_geo(self):
        return derive_world(
            self,
            {QueryType.BBOX},
            self.db.get_bbox_geo,
        )._execute_from_root(QueryType.BBOX)

    def get_time(self):
        return derive_world(
            self,
            {QueryType.BBOX},
            self.db.get_time,
        )._execute_from_root(QueryType.BBOX)

    def _insert_camera(self, camera: Camera):
        return derive_world(
            self,
            {QueryType.CAM},
            self.db.insert_cam,
            camera=camera,
        )

    def _retrieve_camera(self, camera_id: str):
        return derive_world(
            self,
            {QueryType.CAM},
            self.db.retrieve_cam,
            camera_id=camera_id,
        )

    def _insert_bbox_traj(self, camera: Camera, annotation):
        return derive_world(
            self,
            {QueryType.TRAJ, QueryType.BBOX},
            self.db.insert_bbox_traj,
            camera=camera,
            annotation=annotation,
        )

    def _retrieve_bbox(self, camera_id: str):
        return derive_world(self, {QueryType.BBOX}, self.db.retrieve_bbox, camera_id=camera_id)

    def _retrieve_traj(self, camera_id: str):
        return derive_world(self, {QueryType.TRAJ}, self.db.retrieve_traj, camera_id=camera_id)

    def _execute_from_root(self, _type: QueryType):
        nodes: list[World] = []
        curr: Optional[World] = self
        res = None
        query = ""

        if _type is QueryType.CAM:
            query = SnowflakeQuery.from_(Table("cameras")).select("*")
        elif _type is QueryType.BBOX:
            query = SnowflakeQuery.from_(Table("general_bbox")).select("*")
        elif _type is QueryType.TRAJ:
            query = SnowflakeQuery.from_(Table("item_general_trajectory")).select("*")
        else:
            query = ""

        # collect all the nodes til the root
        while curr:
            nodes.append(curr)
            curr = curr._parent

        # execute the nodes from the root
        for node in nodes[::-1]:
            # root
            if node.fn is None:
                continue

            if node.fn == self.db.insert_cam or node.fn == self.db.insert_bbox_traj:
                print("execute:", node.fn.__name__)
                if not node.done:
                    node._execute()
                    node._done = True
            # if different type => pass
            elif _type not in node.types:
                continue
            # treat update method differently
            else:
                print("execute:", node.fn.__name__)
                # print(query)
                query = node._execute(query=query)
        print("done execute node")

        res = query
        # print(query)
        return res

    def _execute(self, **kwargs):
        fn = self._fn[0]
        if fn is None:
            raise Exception("A world without a function should not be executed")

        fn_spec = inspect.getfullargspec(fn)
        if "world_id" in fn_spec.args or fn_spec.varkw is not None:
            return fn(**{"world_id": self._world_id, **self._kwargs, **kwargs})
        return fn(**{**self._kwargs, **kwargs})

    def _print_lineage(self):
        curr = self
        while curr:
            print(curr)
            curr = curr._parent

    def __str__(self):
        return f"fn={self._fn[0]}\nkwargs={self._kwargs}\ndone={self._done}\nworld_id={self._world_id}\n"

    @property
    def world_id(self):
        return self._world_id

    @property
    def timestamp(self):
        return self._timestamp

    @property
    def parent(self):
        return self._parent

    @property
    def name(self):
        return self._name

    @property
    def fn(self):
        return self._fn[0]

    @property
    def kwargs(self):
        return self._kwargs

    @property
    def done(self):
        return self._done

    @property
    def types(self):
        return self._types

    @property
    def materialized(self):
        return self._materialized


def empty_world(name: str) -> World:
    world_id = str(uuid.uuid4())
    timestamp = datetime.datetime.utcnow()
    return World(world_id, timestamp, name)


def derive_world(parent: World, types: set[QueryType], fn: Any, **kwargs) -> World:
    world_id = str(uuid.uuid4())
    timestamp = datetime.datetime.utcnow()

    return World(
        world_id,
        timestamp,
        fn=fn,
        kwargs=kwargs,
        parent=parent,
        types=types,
    )


def trajectory_to_timestamp(trajectory):
    return [traj[0][0]["datetimes"] for traj in trajectory]


def get_overlay_info(trajectory, camera_info: List[List[List["FetchCameraTuple"]]]):
    """
    overlay each trajectory 3d coordinate on to the frame specified by the camera_info
    1. for each trajectory, get the 3d coordinate
    2. get the camera_info associated to it
    3. implement the transformation function from 3d to 2d
        given the single centroid point and camera configuration
        refer to TODO in "senic_utils.py"
    4. return a list of (2d coordinate, frame name/filename)
    """
    result: List[List[List[Tuple[np.ndarray, int, str]]]] = []
    for traj_num in range(len(trajectory)):
        traj_obj = trajectory[traj_num][0]  # traj_obj means the trajectory of current object
        traj_obj_3d = traj_obj["coordinates"]  # 3d coordinate list of the object's trajectory
        camera_info_objs = camera_info[traj_num]  # camera info list corresponding the 3d coordinate
        traj_obj_2d: List[List[Tuple[np.ndarray, int, str]]] = []  # 2d coordinate list
        for index in range(len(camera_info_objs)):
            cur_camera_infos = camera_info_objs[
                index
            ]  # camera info of the obejct in one point of the trajectory
            centroid_3d = np.array(traj_obj_3d[index])  # one point of the trajectory in 3d
            # in order to fit into the function transformation, we develop a dictionary called camera_config
            frame_traj_obj_2d: List[Tuple[np.ndarray, int, str]] = []
            for cur_camera_info in cur_camera_infos:
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
                frame_traj_obj_2d.append((traj_2d, framenum, filename))
            traj_obj_2d.append(frame_traj_obj_2d)
        result.append(traj_obj_2d)
    return result
