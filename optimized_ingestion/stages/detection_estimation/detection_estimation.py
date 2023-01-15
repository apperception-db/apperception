"""Detection Estimation Module

This module is responsible for estimating the object detection throughout the whole video.
The sampling algorithm skips frames based on the current frame geo information.
We estimate objects' metadata only based on the sampled frames.

Usage example:
    from detection_estimation import detection_estimation
    detection_estimation(sorted_ego_config, video, start_frame_num, view_distance=50, img_base_dir='')

TODO:
    1. incoporate yolo detection, either merge this module to the tracking pipeline
    or call yolo detection in this module
    2. ssave the detection and tracking result in the sample plan object

"""

import datetime
import os
import shapely.geometry
import sys
from dataclasses import dataclass, field
from typing import Any, List, Literal, NamedTuple, Tuple

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir)))

import numpy as np
import numpy.typing as npt

from ...camera_config import CameraConfig
from ...types import DetectionId
from ...video import Video
from .sample_plan_algorithms import Action, get_sample_action_alg
from .segment_mapping import CameraPolygonMapping, RoadPolygonInfo
from .utils import (Float2, Float3, Float22, compute_area, compute_distance,
                    detection_to_img_segment, detection_to_nearest_segment,
                    get_largest_segment, get_segment_line,
                    relative_direction_to_ego, trajectory_3d)


class obj_detection(NamedTuple):
    detection_id: DetectionId
    car_loc3d: "Float3"
    car_loc2d: "Float2"
    car_bbox3d: "Tuple[Float3, Float3]"
    car_bbox2d: "Float22"


@dataclass
class DetectionInfo:
    detection_id: "DetectionId"
    frame_segment: "list[npt.NDArray[np.floating]]"
    road_polygon_info: "RoadPolygonInfo"
    car_loc3d: "Float3"
    car_loc2d: "Float2"
    car_bbox3d: "Tuple[Float3, Float3]"
    car_bbox2d: "Float22"
    ego_trajectory: "list[trajectory_3d]"
    ego_config: "CameraConfig"
    ego_road_polygon_info: "RoadPolygonInfo"
    timestamp: "datetime.datetime" = field(init=False)
    road_type: str = field(init=False)
    distance: float = field(init=False)
    segment_area_2d: float = field(init=False)
    relative_direction: "Literal['same_direction', 'opposite_direction']" = field(init=False)
    priority: float = field(init=False)
    segment_line: "shapely.geometry.LineString" = field(init=False)
    segment_heading: float = field(init=False)

    def __post_init__(self):
        timestamp = self.ego_config.timestamp
        assert isinstance(timestamp, datetime.datetime)
        self.timestamp = timestamp
        self.road_type = self.road_polygon_info.road_type

        self.compute_geo_info()
        self.compute_priority()

    def compute_geo_info(self):
        if self.ego_config is None:
            ego_heading = 0
            self.distance = 0
        else:
            ego_heading = self.ego_config.ego_heading
            self.distance = compute_distance(
                self.car_loc3d,
                self.ego_config.ego_translation
            )
        assert isinstance(ego_heading, float)

        if self.car_bbox2d is None:
            self.segment_area_2d = 0
        else:
            self.segment_area_2d = compute_area([
                *self.car_bbox2d[0],
                *self.car_bbox2d[1],
            ])

        self.segment_line, self.segment_heading = get_segment_line(
            self.road_polygon_info,
            self.car_loc3d
        )

        self.relative_direction = relative_direction_to_ego(
            self.segment_heading,
            ego_heading
        )

    def compute_priority(self):
        self.priority = self.segment_area_2d / self.distance

    def generate_single_sample_action(self, view_distance: float = 50.):
        """Generate a sample plan for the given detection of a single car

        Condition 1: detected car is driving towards ego, opposite direction
        Condition 2: detected car driving along the same direction as ego
        Condition 3: detected car and ego are driving into each other
                    i.e. one to north, one from west to it
        Condition 4: detected car and ego are driving away from each other
                    i.e. one to north, one from it to east
        Return: a list of actions
        """
        sample_action_alg = get_sample_action_alg(self.relative_direction)
        if sample_action_alg is not None:
            return self.priority, sample_action_alg(self, view_distance)
        return self.priority, None


@dataclass
class SamplePlan:
    video: "Video"
    next_frame_num: int
    all_detection_info: "List[DetectionInfo]"
    metadata: Any = None
    current_priority: "float | None" = None
    action: "Action | None" = None

    def generate_sample_plan(self, view_distance: float = 50.0):
        assert self.all_detection_info is not None
        for detection_info in self.all_detection_info:
            priority, sample_action = detection_info.generate_single_sample_action(view_distance)
            if sample_action is not None:
                self.add(priority, sample_action)

    def add(self, priority: float, sample_action: "Action", time_threshold: float = 0.5):
        assert sample_action is not None
        if sample_action.invalid_action:
            return
        # assert not sample_action.invalid_action

        assert (self.action is None) == (self.current_priority is None)
        if self.action is None or self.current_priority is None:
            self.current_priority = priority
            self.action = sample_action
        else:
            if sample_action.estimated_time < self.action.estimated_time:
                if (priority >= self.current_priority
                    or sample_action.estimated_time / self.action.estimated_time
                        < time_threshold):
                    self.current_priority = priority
                    self.action = sample_action

    def get_action(self):
        return self.action

    def get_action_type(self):
        if self.action is None:
            return None
        return self.action.action_type

    def get_next_sample_frame_info(self):
        if self.action is None:
            return None

        nearest_index = None
        min_diff = None
        for i, config in enumerate(self.video):
            timestamp = config.timestamp
            diff = self.action.finish_time - timestamp
            if diff.total_seconds() < 0:
                diff = -diff
            if min_diff is None or min_diff > diff:
                min_diff = diff
                nearest_index = i

        return None, nearest_index, None

    def get_next_frame_num(self, next_frame_num: int):
        next_sample_frame_info = self.get_next_sample_frame_info()
        if next_sample_frame_info:
            _, next_sample_frame_num, _ = next_sample_frame_info
            self.next_frame_num = max(next_sample_frame_num, next_frame_num)
        return self.next_frame_num


def construct_all_detection_info(
    cam_polygon_mapping: "List[CameraPolygonMapping]",
    ego_config: "CameraConfig",
    ego_trajectory: "list[trajectory_3d]",
    all_detections: "list[obj_detection]"
):
    all_detection_info: "List[DetectionInfo]" = []
    if len(all_detections) == 0:
        return all_detection_info
    ego_mapping = get_largest_segment(cam_polygon_mapping)
    if ego_mapping is None:
        # for mapping in cam_segment_mapping:
        #     cam_segment, road_segment_info = mapping
        raise ValueError('Ego segment not included')

    _, ego_road_segment_info = ego_mapping

    for detection in all_detections:
        detection_id, car_loc3d, car_loc2d, car_bbox3d, car_bbox2d = detection
        related_mapping = detection_to_img_segment(car_loc2d, cam_polygon_mapping)
        if related_mapping is None:
            cam_segment, road_segment_info = detection_to_nearest_segment(car_loc3d, cam_polygon_mapping)
        else:
            cam_segment, road_segment_info = related_mapping

        detection_info = DetectionInfo(detection_id,
                                       cam_segment,
                                       road_segment_info,
                                       car_loc3d,
                                       car_loc2d,
                                       car_bbox3d,
                                       car_bbox2d,
                                       ego_trajectory,
                                       ego_config,
                                       ego_road_segment_info)
        all_detection_info.append(detection_info)

    return all_detection_info


def generate_sample_plan(
    video: "Video",
    next_frame_num: int,
    all_detection_info: "List[DetectionInfo]",
    view_distance: float,
):
    ### the object detection with higher priority doesn't necessarily get sampled first,
    # it also based on the sample plan
    sample_plan = SamplePlan(video, next_frame_num, all_detection_info)
    sample_plan.generate_sample_plan(view_distance)
    return sample_plan
