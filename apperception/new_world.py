from __future__ import annotations

import datetime
import glob
import inspect
import string
import uuid
from collections.abc import Iterable
from enum import IntEnum
from os import makedirs, path
from pyclbr import Function
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import dill as pickle
import matplotlib
import numpy as np
import yaml
from camera import Camera
from new_db import Database
from new_util import compile_lambda
from pypika import Table
from pypika.dialects import SnowflakeQuery
from scenic_util import transformation

# matplotlib.use("Qt5Agg")
# print("get backend", matplotlib.get_backend())

makedirs("./.apperception_cache", exist_ok=True)


class Type(IntEnum):
    # query type: for example, if we call get_cam(), and we execute the commands from root. when we encounter
    # recognize(), we should not execute it because the inserted object must not be in the final result. we use enum
    # type to determine whether we should execute this node
    CAM, BBOX, TRAJ = 0, 1, 2


BASE_VOLUME_QUERY_TEXT = "STBOX Z(({x1}, {y1}, {z1}),({x2}, {y2}, {z2}))"


class World:
    # all worlds share a db instance
    db = Database()
    camera_nodes: Dict[str, Camera] = {}

    _parent: Optional[World]
    _name: str
    # TODO: Fix _fn typing: (World, *Any, **Any) -> Query | str? | None
    _fn: Any
    _kwargs: dict[str, Any]
    _done: bool
    _world_id: str
    _timestamp: datetime.datetime
    _types: set[Type]
    _materialized: bool

    def __init__(
        self,
        world_id: str,
        timestamp: datetime.datetime,
        name: str = None,
        parent: World = None,
        fn: Any = None,
        kwargs: dict[str, Any] = None,
        done: bool = False,
        types: Set[Type] = None,
        materialized: bool = False,
    ):
        self._parent = parent
        self._name = "" if name is None else name
        self._fn = None if fn is None else (getattr(self.db, fn) if isinstance(fn, str) else fn)
        self._kwargs = {} if kwargs is None else kwargs
        self._done = done  # update node
        self._world_id = world_id
        self._timestamp = timestamp
        self._types = set() if types is None else types
        self._materialized = materialized

    def road_direction(self, x, y):
        return self.db.get_heading_from_a_point(x, y)

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

    def select_by_range(self, cam_id, x_range: Tuple[float, float], z_range: Tuple[float, float]):
        camera = self.camera_nodes[cam_id]
        x, _, z = camera.point.coordinate

        x_min = x + x_range[0]
        x_max = x + x_range[1]
        z_min = z + z_range[0]
        z_max = z + z_range[1]

        return BASE_VOLUME_QUERY_TEXT.format(
            x1=x_min, y1=float("-inf"), z1=z_min, x2=x_max, y2=float("inf"), z2=z_max
        )

    def overlay_trajectory(self, scene_name: string, trajectory, object_id: string):
        frame_num = self.trajectory_to_frame_num(trajectory)
        # frame_num is int[[]], hence camera_info should also be [[]]
        camera_info = []  # camera_info is a list of mappings from frameNum to list of cameras
        for index, cur_frame_num in enumerate(frame_num):
            current_cameras = self.db.fetch_camera(scene_name, cur_frame_num)
            camera_info.append({})
            for x in current_cameras:
                if x[6] in camera_info[index]:
                    camera_info[index][x[6]].append(x)
                else:
                    camera_info[index][x[6]] = [x]
        camera_info = [
            [x[y] for y in sorted(x)] for x in camera_info
        ]  # [x.values() for x in sorted(camera_info, key=lambda x: x[6])]

        # assert len(camera_info) == len(frame_num)
        # assert len(camera_info[0]) == len(frame_num[0])
        # print(camera_info, np.asarray(camera_info).shape) # (1, 30, 8)
        # print(trajectory, np.asarray(trajectory).shape) # (1, 1)
        overlay_info = self.get_overlay_info(trajectory, camera_info)
        # TODO: fix the following to overlay the 2d point onto the frame
        results = []
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

    def trajectory_to_frame_num(self, trajectory):
        """
        fetch the frame number from the trajectory
        1. get the time stamp field from the trajectory
        2. convert the time stamp to frame number
            Refer to 'convert_datetime_to_frame_num' in 'video_util.py'
        3. return the frame number
        """
        frame_num = []
        start_time = self.db.start_time
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
            camera_info_objs = camera_info[
                traj_num
            ]  # camera info list corresponding the 3d coordinate
            traj_obj_2d = []  # 2d coordinate list
            for index in range(len(camera_info_objs)):
                cur_camera_infos = camera_info_objs[
                    index
                ]  # camera info of the obejct in one point of the trajectory
                centroid_3d = np.array(traj_obj_3d[index])  # one point of the trajectory in 3d
                # in order to fit into the function transformation, we develop a dictionary called camera_config
                frame_traj_obj_2d = []
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

    def recognize(self, camera: Camera, annotation):
        node1 = self._insert_bbox_traj(camera=camera, annotation=annotation)
        node2 = node1._retrieve_bbox(camera_id=camera.id)
        node3 = node2._retrieve_traj(camera_id=camera.id)
        return node3

    def get_video(self, cam_ids: List[str] = [], boxed: bool = False):
        return derive_world(
            self,
            {Type.TRAJ},
            self.db.get_video,
            cams=[World.camera_nodes[cam_id] for cam_id in cam_ids],
            boxed=boxed,
        )._execute_from_root(Type.TRAJ)

    def get_bbox(self):
        return derive_world(
            self,
            {Type.BBOX},
            self.db.get_bbox,
        )._execute_from_root(Type.BBOX)

    def get_traj(self):
        return derive_world(
            self,
            {Type.TRAJ},
            self.db.get_traj,
        )._execute_from_root(Type.TRAJ)

    def get_traj_key(self):
        return derive_world(
            self,
            {Type.TRAJ},
            self.db.get_traj_key,
        )._execute_from_root(Type.TRAJ)

    def get_headings(self):
        # TODO: Optimize operations with NumPy if possible
        trajectories = self.get_traj()
        headings = []
        for traj in trajectories:
            traj = traj[0]
            heading = [None]
            for j in range(1, len(traj)):
                prev_pos = traj[j - 1]
                current_pos = traj[j]
                heading.append(0)
                if current_pos[1] != prev_pos[1]:
                    heading[j] = np.arctan2(
                        current_pos[1] - prev_pos[1], current_pos[0] - prev_pos[0]
                    )
                heading[j] *= 180 / np.pi  # convert to degrees from radian
                heading[j] = (
                    heading[j] + 360
                ) % 360  # converting such that all headings are positive
            headings.append(heading)
        return headings

    def get_distance(self, start: float, end: float):
        return derive_world(
            self,
            {Type.TRAJ},
            self.db.get_distance,
            start=str(self.db.start_time + datetime.timedelta(seconds=start)),
            end=str(self.db.start_time + datetime.timedelta(seconds=end)),
        )._execute_from_root(Type.TRAJ)

    def get_speed(self, start, end):
        return derive_world(
            self,
            {Type.TRAJ},
            self.db.get_speed,
            start=str(self.db.start_time + datetime.timedelta(seconds=start)),
            end=str(self.db.start_time + datetime.timedelta(seconds=end)),
        )._execute_from_root(Type.TRAJ)

    def filter_traj_type(self, object_type: str):
        return derive_world(self, {Type.TRAJ}, self.db.filter_traj_type, object_type=object_type)

    def filter_traj_volume(self, volume: str):
        return derive_world(self, {Type.TRAJ}, self.db.filter_traj_volume, volume=volume)

    def filter_traj_heading(self, lessThan=float("inf"), greaterThan=float("-inf")):
        return derive_world(
            self,
            {Type.TRAJ},
            self.db.filter_traj_heading,
            lessThan=lessThan,
            greaterThan=greaterThan,
        )

    def filter_relative_to_type(
        self,
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        z_range: Tuple[float, float],
        type: str,
    ):
        return derive_world(
            self,
            {Type.TRAJ},
            self.db.filter_relative_to_type,
            x_range=x_range,
            y_range=y_range,
            z_range=z_range,
            type=type,
        )

    def filter_pred_relative_to_type(self, pred: Function):
        x_range, y_range = compile_lambda(pred)

        return derive_world(
            self,
            {Type.TRAJ},
            self.db.filter_relative_to_type,
            x_range=x_range,
            y_range=y_range,
            z_range=[float(-(2**31)), float(2**31)],
            type="camera",
        )

    def add_camera(self, camera: Camera):
        """
        1. For update method, we create two nodes: the first node will write to the db, and the second node will retrieve from the db
        2. For the write node, never double write. (so we use done flag)
        ... -> [write] -> [retrive] -> ...
        """
        node1 = self._insert_camera(camera=camera)
        node2 = node1._retrieve_camera(camera_id=camera.id)
        return node2

    def interval(self, start, end):
        return derive_world(
            self,
            {Type.BBOX},
            self.db.interval,
            start=str(self.db.start_time + datetime.timedelta(seconds=start)),
            end=str(self.db.start_time + datetime.timedelta(seconds=end)),
        )

    def add_properties(self, cam_id: str, properties: Any, property_type: str, new_prop):
        # TODO: Should we add this to DB instead of the global object?
        self.camera_nodes[cam_id].add_property(properties, property_type, new_prop)

    def predicate(self, func: Function):

        return derive_world(
            self,
            {Type.TRAJ, Type.BBOX},
            self.db.predicate,
            func=func,
        )

    def get_len(self):
        return derive_world(
            self,
            {Type.CAM},
            self.db.get_len,
        )._execute_from_root(Type.CAM)

    def get_camera(self):
        return derive_world(
            self,
            {Type.CAM},
            self.db.get_cam,
        )._execute_from_root(Type.CAM)

    def get_bbox_geo(self):
        return derive_world(
            self,
            {Type.BBOX},
            self.db.get_bbox_geo,
        )._execute_from_root(Type.BBOX)

    def get_time(self):
        return derive_world(
            self,
            {Type.BBOX},
            self.db.get_time,
        )._execute_from_root(Type.BBOX)

    def _insert_camera(self, camera: Camera):
        return derive_world(
            self,
            {Type.CAM},
            self.db.insert_cam,
            camera=camera,
        )

    def _retrieve_camera(self, camera_id: str):
        return derive_world(
            self,
            {Type.CAM},
            self.db.retrieve_cam,
            camera_id=camera_id,
        )

    def _insert_bbox_traj(self, camera: Camera, annotation):
        return derive_world(
            self,
            {Type.TRAJ, Type.BBOX},
            self.db.insert_bbox_traj,
            camera=camera,
            annotation=annotation,
        )

    def _retrieve_bbox(self, camera_id: str):
        return derive_world(self, {Type.BBOX}, self.db.retrieve_bbox, camera_id=camera_id)

    def _retrieve_traj(self, camera_id: str):
        return derive_world(self, {Type.TRAJ}, self.db.retrieve_traj, camera_id=camera_id)

    def _execute_from_root(self, type: Type):
        nodes: list[World] = []
        curr: Optional[World] = self
        res = None
        query = ""

        if type is Type.CAM:
            query = SnowflakeQuery.from_(Table("cameras")).select("*")
        elif type is Type.BBOX:
            query = SnowflakeQuery.from_(Table("general_bbox")).select("*")
        elif type is Type.TRAJ:
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
                    node._update_log_file()
            # if different type => pass
            elif type not in node.types:
                continue
            # treat update method differently
            else:
                print("execute:", node.fn.__name__)
                # print(query)
                query = node._execute(query=query)
        print("done execute node")

        res = query
        return res

    def _execute(self, **kwargs):
        fn_spec = inspect.getfullargspec(self._fn)
        if "world_id" in fn_spec.args or fn_spec.varkw is not None:
            return self._fn(**{"world_id": self._world_id, **self._kwargs, **kwargs})
        return self._fn(**{**self._kwargs, **kwargs})

    def _print_lineage(self):
        curr = self
        while curr:
            print(curr)
            curr = curr._parent

    def __str__(self):
        return (
            f"fn={self._fn}\nkwargs={self._kwargs}\ndone={self._done}\nworld_id={self._world_id}\n"
        )

    @property
    def filename(self):
        return filename(self._timestamp, self._world_id, self._name)

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
        return self._fn

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

    def _update_log_file(self):
        # with open(self.filename, "r") as f:
        #     children = yaml.safe_load(f).get("children_filenames", None)
        # with open(self.filename, "w") as f:
        #     f.write(
        #         yaml.safe_dump(
        #             {
        #                 **({} if self._parent is None else {"parent": self._parent.filename}),
        #                 **({} if self._types == set() else {"types": set(map(int, self._types))}),
        #                 **({} if self._fn is None else {"fn": self._fn.__name__}),
        #                 **({} if self._kwargs == {} else {"kwargs": pickle.dumps(self._kwargs)}),
        #                 **({} if not self._done else {"done": self._done}),
        #                 **({} if not self._materialized else {"materialized": self._materialized}),
        #                 **({} if children is None else {"children_filenames": children}),
        #             }
        #         )
        #     )
        pass


def empty_world(name: str) -> World:
    matched_files = list(
        filter(path.isfile, glob.glob(f"./.apperception_cache/*_*_{name}.ap.yaml"))
    )
    if len(matched_files):
        return _empty_world_from_file(matched_files[0])
    return _empty_world(name)


def _empty_world_from_file(log_file: str) -> World:
    with open(log_file, "r") as f:
        content = yaml.safe_load(f)
        if "children_filenames" in content:
            del content["children_filenames"]
        return World(*split_filename(log_file), **content)


def _empty_world(name: str) -> World:
    world_id = str(uuid.uuid4())
    timestamp = datetime.datetime.utcnow()
    log_file = filename(timestamp, world_id, name)
    with open(log_file, "w") as f:
        f.write(yaml.safe_dump({}))
    return World(world_id, timestamp, name)


def derive_world(parent: World, types: set[Type], fn: Any, **kwargs) -> World:
    # world = _derive_world_from_file(parent, types, fn, **kwargs)
    # if world is not None:
    #     return world
    return _derive_world(parent, types, fn, **kwargs)


def _derive_world(parent: World, types: set[Type], fn: Any, **kwargs) -> World:
    world_id = str(uuid.uuid4())
    timestamp = datetime.datetime.utcnow()
    # log_file = filename(timestamp, world_id)

    # with open(parent.filename, "r") as pf:
    #     content = yaml.safe_load(pf)
    # with open(parent.filename, "w") as pf:
    #     content["children_filenames"] = content.get("children_filenames", set())
    #     content["children_filenames"].add(log_file)
    #     pf.write(yaml.safe_dump(content))

    # with open(log_file, "w") as f:
    #     f.write(
    #         yaml.safe_dump(
    #             {
    #                 "fn": fn.__name__,
    #                 "kwargs": pickle.dumps(kwargs),
    #                 "parent": parent.filename,
    #                 "types": set(map(int, types)),
    #             }
    #         )
    #     )

    return World(
        world_id,
        timestamp,
        fn=fn,
        kwargs=kwargs,
        parent=parent,
        types=types,
    )


def _derive_world_from_file(parent: World, types: set[Type], fn: Any, **kwargs) -> Optional[World]:
    with open(parent.filename, "r") as f:
        sibling_filenames: Iterable[str] = yaml.safe_load(f).get("children_filenames", [])

    for sibling_filename in sibling_filenames:
        with open(sibling_filename, "r") as sf:
            sibling_content = yaml.safe_load(sf)

        if op_matched(sibling_content, types, fn, kwargs):
            return World(
                *split_filename(sibling_filename),
                parent=parent,
                **format_content(sibling_content),
            )

    return None


def from_file(filename: str) -> World:
    with open(filename, "r") as f:
        content = yaml.safe_load(f)

    parent_filename = content.get("parent_filename", None)
    if parent_filename is None:
        parent = None
    else:
        parent = from_file(parent_filename)

    return World(*split_filename(filename), parent=parent, **format_content(content))


def filename(timestamp: datetime.datetime, world_id: str, name: str = ""):
    return f".apperception_cache/{str(timestamp).replace(':', ';')}_{world_id}_{name}.ap.yaml"


def split_filename(filename: str) -> Tuple[str, datetime.datetime, str]:
    filename = filename.replace("\\", "/")
    timestamp_str, world_id, name = filename[: -len(".ap.yaml")].split("/")[-1].split("_", 2)
    return world_id, datetime.datetime.fromisoformat(timestamp_str.replace(";", ":")), name


DUMPED_EMPTY_DICT = pickle.dumps({})


def double_equal(a: Tuple[Any, Any]):
    return a[0] == a[1]


def op_matched(
    file_content: dict[str, Any],
    types: set[Type],
    fn: Any,
    kwargs: dict[str, Any] = None,
) -> bool:
    f_fn: str | None = file_content.get("fn", None)
    f_types: set[int] = file_content.get("types", set())

    if f_fn != fn.__name__ or f_types != set(map(int, types)):
        return False

    kwargs = {} if kwargs is None else kwargs
    f_kwargs: dict[str, Any] = pickle.loads(file_content.get("kwargs", DUMPED_EMPTY_DICT))

    if len(f_kwargs) != len(kwargs):
        return False

    cmps = fn.comparators if hasattr(fn, "comparators") else {}
    a = all(
        key in f_kwargs and cmps.get(key, double_equal)((f_kwargs[key], kwargs[key]))
        for key in kwargs
    )
    return a


def format_content(content: dict[str, Any]) -> dict[str, Any]:
    if "types" in content:
        content["types"] = set(map(Type, content["types"]))

    if "kwargs" in content:
        content["kwargs"] = pickle.loads(content["kwargs"])

    if "parent" in content:
        del content["parent"]

    if "children_filenames" in content:
        del content["children_filenames"]

    return content
