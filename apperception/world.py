from __future__ import annotations

import datetime
import inspect
import uuid
from typing import (TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple,
                    Union)

import cv2
import numpy as np
from pypika import Table
from pypika.dialects import SnowflakeQuery

from apperception.data_types import Camera, FetchCameraTuple
from apperception.database import database
from apperception.utils import transformation

if TYPE_CHECKING:
    import pandas as pd

    from .data_types import Trajectory


camera_nodes: Dict[str, "Camera"] = {}


class World:
    _parent: Optional[World]
    _name: str
    _fn: Tuple[Optional[Callable]]
    _kwargs: dict[str, Any]
    _done: bool
    _world_id: str
    _timestamp: datetime.datetime
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
        materialized: bool = False,
    ):
        self._parent = parent
        self._name = "" if name is None else name
        self._fn = (fn if fn is None else (getattr(database, fn) if isinstance(fn, str) else fn),)
        self._kwargs = {} if kwargs is None else kwargs
        self._done = done  # update node
        self._world_id = world_id
        self._timestamp = timestamp
        self._materialized = materialized

    def get_trajectory_images(
        self,
        scene_name: str,
        trajectory: Trajectory,
    ):
        frame_nums = database.timestamp_to_framenum(
            scene_name, ["'" + x + "'" for x in trajectory.datetimes]
        )
        # c amera_info is a list of list of cameras, where the list of cameras at each index represents the cameras at the respective timestamp
        image_names: List[List[str]] = []
        for frame_num in frame_nums:
            current_cameras = database.fetch_camera_framenum(scene_name, [frame_num[0]])
            currant_images = [x[7] for x in current_cameras]
            image_names.append(currant_images)
        return image_names

    # TODO: Eventually move these to there own utils file.s
    def overlay_trajectory(
        self,
        scene_name: str,
        trajectory: Trajectory,
        file_name: str,
        overlay_headings: bool = False,
        overlay_road: bool = False,
    ):
        frame_nums = database.timestamp_to_framenum(
            scene_name, ["'" + x + "'" for x in trajectory.datetimes]
        )
        # c amera_info is a list of list of cameras, where the list of cameras at each index represents the cameras at the respective timestamp
        camera_info: List[List["FetchCameraTuple"]] = []
        for frame_num in frame_nums:
            current_cameras = database.fetch_camera_framenum(scene_name, [frame_num[0]])
            camera_info.append(current_cameras)

        overlay_info = get_overlay_info(trajectory, camera_info)

        for file_prefix in overlay_info:
            frame_width = None
            frame_height = None
            vid_writer = None
            camera_points = overlay_info[file_prefix]
            for point in camera_points:
                (
                    traj_2d,
                    framenum,
                    filename,
                    camera_heading,
                    ego_heading,
                    ego_translation,
                    camera_config,
                ) = point
                frame_im = cv2.imread(filename)
                cv2.circle(
                    frame_im,
                    tuple([int(traj_2d[0][0]), int(traj_2d[1][0])]),
                    10,
                    (0, 255, 0),
                    -1,
                )
                if overlay_headings:
                    cam_road_dir = self.road_direction(ego_translation[0], ego_translation[1])[0][0]
                    stats = {
                        "Ego Heading": str(round(ego_heading, 2)),
                        "Camera Heading": str(round(camera_heading, 2)),
                        "Road Direction": str(round(cam_road_dir, 2)),
                    }
                    self.overlay_stats(frame_im, stats)
                if overlay_road:
                    frame_im = self.overlay_road(frame_im, camera_config)
                if vid_writer is None:
                    frame_height, frame_width = frame_im.shape[:2]
                    vid_writer = cv2.VideoWriter(
                        "./output/" + file_name + "." + file_prefix + ".mp4",
                        cv2.VideoWriter_fourcc("m", "p", "4", "v"),
                        10,
                        (frame_width, frame_height),
                    )
                vid_writer.write(frame_im)
            if vid_writer is not None:
                vid_writer.release()

    def overlay_stats(self, frame, stats):
        x, y = 10, 50
        for stat in stats:
            statValue = stats[stat]
            cv2.putText(
                frame,
                stat + ": " + statValue,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                3,
            )
            y += 50

    def overlay_road(self, frame, camera_config):
        ego_translation = camera_config["egoTranslation"]
        camera_road_coords = self.road_coords(ego_translation[0], ego_translation[1])[0][0]
        pixel_start_enc = world_to_pixel(
            camera_config, (camera_road_coords[0], camera_road_coords[1], 0)
        )
        pixel_end_enc = world_to_pixel(
            camera_config, (camera_road_coords[2], camera_road_coords[3], 0)
        )
        pixel_start = tuple([int(pixel_start_enc[0][0]), int(pixel_start_enc[1][0])])
        pixel_end = tuple([int(pixel_end_enc[0][0]), int(pixel_end_enc[1][0])])
        print(camera_road_coords, ego_translation)
        frame = cv2.line(
            frame,
            pixel_start,
            pixel_end,
            (255, 0, 0),
            3,
        )
        return frame

    def add_camera(self, camera: "Camera"):
        node1 = self._insert_camera(camera=camera)
        node2 = node1._retrieve_camera(camera_id=camera.id)
        return node2

    def recognize(self, camera: "Camera", annotation: "pd.DataFrame"):
        node1 = self._insert_bbox_traj(camera=camera, annotation=annotation)
        node2 = node1._retrieve_bbox(camera_id=camera.id)
        node3 = node2._retrieve_traj(camera_id=camera.id)
        return node3

    def filter(self, predicate: Union[str, Callable]) -> World:
        return derive_world(
            self,
            database.filter,
            predicate=predicate,
        )

    def exclude(self, other: World) -> World:
        return derive_world(self, database.exclude, world=other)

    def union(self, other: World) -> World:
        return derive_world(self, database.union, world=other)

    def intersect(self, other: World) -> World:
        return derive_world(self, database.intersect, world=other)

    def sym_diff(self, other: World) -> World:
        return self.union(other).exclude(self.intersect(other))

    def __lshift__(self, camera: Union["Camera", Tuple["Camera", "pd.DataFrame"]]):
        """add a camera or add a camera and recognize"""
        if isinstance(camera, Camera):
            return self.add_camera(camera)
        c, a, *_ = camera
        return self.add_camera(c).recognize(c, a)

    def __sub__(self, other: World) -> World:
        return self.exclude(other)

    def __or__(self, other: World) -> World:
        return self.union(other)

    def __and__(self, other: World) -> World:
        return self.intersect(other)

    def __xor__(self, other: World) -> World:
        return self.sym_diff(other)

    def select_all(self):
        return derive_world(self, database.select_all)._execute_from_root()

    def get_video(self, cam_ids: List[str] = [], boxed: bool = False):
        return derive_world(
            self,
            database.get_video,
            cams=[camera_nodes[cam_id] for cam_id in cam_ids],
            boxed=boxed,
        )._execute_from_root()

    def road_direction(self, x: float, y: float):
        return database.road_direction(x, y)

    def road_coords(self, x: float, y: float):
        return database.road_coords(x, y)

    def get_traj(self) -> List[List["Trajectory"]]:
        return derive_world(
            self,
            database.get_traj,
        )._execute_from_root()

    def get_traj_key(self):
        return derive_world(
            self,
            database.get_traj_key,
        )._execute_from_root()

    def get_traj_attr(self, attr: str):
        return derive_world(self, database.get_traj_attr, attr=attr)._execute_from_root()

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
            database.get_distance,
            start=str(start),
            end=str(end),
        )._execute_from_root()

    def get_speed(self, start: datetime.datetime, end: datetime.datetime):
        return derive_world(
            self,
            database.get_speed,
            start=str(start),
            end=str(end),
        )._execute_from_root()

    def get_len(self):
        return derive_world(
            self,
            database.get_len,
        )._execute_from_root()

    def get_camera(self):
        return derive_world(
            self,
            database.get_cam,
        )._execute_from_root()

    def get_bbox_geo(self):
        return derive_world(
            self,
            database.get_bbox_geo,
        )._execute_from_root()

    def get_time(self):
        return derive_world(
            self,
            database.get_time,
        )._execute_from_root()

    def get_id_time_camId_filename(self, num_joined_tables: int):
        return derive_world(
            self,
            database.get_id_time_camId_filename,
            num_joined_tables=num_joined_tables,
        )._execute_from_root()

    def _insert_camera(self, camera: "Camera"):
        return derive_world(
            self,
            database.insert_cam,
            camera=camera,
        )

    def _retrieve_camera(self, camera_id: str):
        return derive_world(
            self,
            database.retrieve_cam,
            camera_id=camera_id,
        )

    def _insert_bbox_traj(self, camera: "Camera", annotation: "pd.DataFrame"):
        return derive_world(
            self,
            database.insert_bbox_traj,
            camera=camera,
            annotation=annotation,
        )

    def _retrieve_bbox(self, camera_id: str):
        return derive_world(self, database.retrieve_bbox, camera_id=camera_id)

    def _retrieve_traj(self, camera_id: str):
        return derive_world(self, database.retrieve_traj, camera_id=camera_id)

    def _execute_from_root(self) -> Any:
        nodes: list[World] = []
        curr: Optional[World] = self
        res = None
        query = SnowflakeQuery.from_(Table("item_general_trajectory")).select("*")

        # collect all the nodes til the root
        while curr:
            nodes.append(curr)
            curr = curr._parent

        # execute the nodes from the root
        for node in nodes[::-1]:
            # root
            if node.fn is None:
                continue

            if node.fn == database.insert_cam or node.fn == database.insert_bbox_traj:
                print("execute:", node.fn.__name__)
                if not node.done:
                    node._execute()
                    node._done = True
            # treat update method differently
            else:
                print("execute:", node.fn.__name__)
                # print(query)
                query = node._execute(query=query, **node.kwargs)
        print("done execute node")

        res = query
        if isinstance(query, list):
            print("Result length:", len(query))
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
    def materialized(self):
        return self._materialized


def empty_world(name: str) -> World:
    world_id = str(uuid.uuid4())
    timestamp = datetime.datetime.utcnow()
    return World(world_id, timestamp, name)


def derive_world(parent: World, fn: Any, **kwargs) -> World:
    world_id = str(uuid.uuid4())
    timestamp = datetime.datetime.utcnow()

    return World(
        world_id,
        timestamp,
        fn=fn,
        kwargs=kwargs,
        parent=parent,
    )


def world_to_pixel(
    camera_config: dict, world_coords: Union[np.ndarray, Tuple[float, float, float]]
):
    traj_2d = transformation(world_coords, camera_config)
    return traj_2d


def get_overlay_info(trajectory: Trajectory, camera_info: List[List["FetchCameraTuple"]]):
    """
    overlay each trajectory 3d coordinate on to the frame specified by the camera_info
    1. For each point in the trajectory, find the list of cameras that correspond to that timestamp
    2. Project the trajectory coordinates onto the intrinsics of the camera, and add it to the list of results
    3. Returns a mapping from each camera type (FRONT, BACK, etc) to the trajectory in pixel coordinates of that camera
    """
    traj_obj_3d = trajectory.coordinates
    result: Dict[str, List[Tuple[np.ndarray, int, str, float, float, List[float], dict]]] = {}
    for index, cur_camera_infos in enumerate(camera_info):
        # cur_camera_infos = camera_info[
        #     index
        # ]  # camera info of the obejct in one point of the trajectory
        centroid_3d = np.array(traj_obj_3d[index])  # one point of the trajectory in 3d
        # in order to fit into the function transformation, we develop a dictionary called camera_config
        for cur_camera_info in cur_camera_infos:
            # TODO: add type to camera_config
            camera_config: Dict[str, Any] = {}
            camera_config["egoTranslation"] = cur_camera_info[1]
            camera_config["egoRotation"] = np.array(cur_camera_info[2])
            camera_config["cameraTranslation"] = cur_camera_info[3]
            camera_config["cameraRotation"] = np.array(cur_camera_info[4])
            camera_config["cameraIntrinsic"] = np.array(cur_camera_info[5])

            traj_2d = world_to_pixel(camera_config, centroid_3d)

            framenum = cur_camera_info[6]
            filename = cur_camera_info[7]
            camera_heading = cur_camera_info[8]
            ego_heading = cur_camera_info[9]
            ego_translation = cur_camera_info[1]
            file_prefix = "_".join(filename.split("/")[:-1])

            if file_prefix not in result:
                result[file_prefix] = []
            result[file_prefix].append(
                (
                    traj_2d,
                    framenum,
                    filename,
                    camera_heading,
                    ego_heading,
                    ego_translation,
                    camera_config,
                )
            )
    return result


def trajectory_to_timestamp(trajectory):
    return [traj[0].datetimes for traj in trajectory]
