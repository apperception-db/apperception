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
import sys
from dataclasses import dataclass, field
from typing import Any, List, Literal, Tuple

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir)))

import shapely

from ...camera_config import CameraConfig
from ...types import DetectionId, obj_detection
from ...video import Video
from .optimized_segment_mapping import (
    RoadPolygonInfo,
    get_detection_polygon_mapping,
    get_largest_polygon_containing_point,
)
from .sample_plan_algorithms import Action, CAR_EXIT_SEGMENT
from .utils import (
    Float2,
    Float3,
    Float22,
    compute_area,
    compute_distance,
    get_car_exits_view_frame_num,
    get_segment_line,
    relative_direction_to_ego,
    time_to_exit_current_segment,
    trajectory_3d,
)


@dataclass
class DetectionInfo:
    detection_id: "DetectionId"
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
        # if self.segment_line is None:
        #     print(self.road_polygon_info.id)
        #     print(self.road_type)
        # if self.detection_id.frame_idx == 55 and self.road_type != 'intersection':
        #     print([self.detection_id.obj_order, self.car_bbox2d], ',')
        #     relative_heading = abs(self.segment_heading - ego_heading) % 360
        # print(self.detection_id)
        # print(self.road_type)
        # print(math.cos(math.radians(relative_heading)))

    def compute_priority(self):
        self.priority = self.segment_area_2d / self.distance

    def get_car_exits_segment_action(self):
        current_time = self.timestamp
        car_loc = self.car_loc3d
        exit_time, exit_point = time_to_exit_current_segment(self, current_time, car_loc)
        if exit_time is None or exit_point is None:
            return None
        return Action(current_time, exit_time, start_loc=car_loc,
                      end_loc=exit_point, action_type=CAR_EXIT_SEGMENT,
                      target_obj_id=self.detection_id,
                      target_obj_bbox=self.car_bbox2d)

    # def generate_single_sample_action(self, view_distance: float = 50.):
    #     """Generate a sample plan for the given detection of a single car

    #     Condition 1: detected car is driving towards ego, opposite direction
    #     Condition 2: detected car driving along the same direction as ego
    #     Condition 3: detected car and ego are driving into each other
    #                 i.e. one to north, one from west to it
    #     Condition 4: detected car and ego are driving away from each other
    #                 i.e. one to north, one from it to east
    #     Return: a list of actions
    #     """
    #     sample_action_alg = get_sample_action_alg(self.relative_direction)
    #     if sample_action_alg is not None:
    #         return self.priority, sample_action_alg(self, view_distance)
    #     return self.priority, None


@dataclass
class SamplePlan:
    video: "Video"
    next_frame_num: int
    all_detection_info: "List[DetectionInfo]"
    ego_views: "List[shape.geometry.Polygon]"
    fps: int = 20
    metadata: Any = None
    current_priority: "float | None" = None
    action: "Action | None" = None

    def update_next_sample_frame_num(self):
        next_sample_frame_num = len(self.ego_views)
        assert self.all_detection_info is not None
        for detection_info in self.all_detection_info:
            # get the frame num of car exits,
            # if the detection_info road type is intersection
            # car_exit_segment_action would be None
            car_exit_segment_action = detection_info.get_car_exits_segment_action()
            if car_exit_segment_action and not car_exit_segment_action.invalid_action:
                car_exit_segment_frame_num = self.find_closest_frame_num(
                    car_exit_segment_action.finish_time)
                if car_exit_segment_frame_num > self.next_frame_num:
                    # get the frame num of car exits view
                    car_exit_view_frame_num = get_car_exits_view_frame_num(
                        detection_info, self.ego_views, car_exit_segment_frame_num, self.fps)
                    next_sample_frame_num = min(next_sample_frame_num,
                                                car_exit_segment_frame_num,
                                                car_exit_view_frame_num)
                else:
                    next_sample_frame_num = self.next_frame_num
                    break
            else:
                next_sample_frame_num = self.next_frame_num
                break
        self.next_frame_num = next_sample_frame_num

    def generate_sample_plan(self, view_distance: float = 50.0):
        assert self.all_detection_info is not None
        for detection_info in self.all_detection_info:
            priority, sample_action = detection_info.generate_single_sample_action(view_distance)
            if sample_action is not None:
                self.add(priority, sample_action)
        if self.action and not self.action.invalid_action:
            self.next_frame_num = min(
                self.next_frame_num,
                self.find_closest_frame_num(self.action.finish_time))

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

    def find_closest_frame_num(self, finish_time: "datetime.datetime"):
        if finish_time is None:
            return None

        nearest_index = None
        min_diff = None
        for i, config in enumerate(self.video):
            timestamp = config.timestamp
            if timestamp > finish_time:
                break
            diff = finish_time - timestamp
            assert diff.total_seconds() >= 0
            if min_diff is None or min_diff > diff:
                min_diff = diff
                nearest_index = i

        return nearest_index

    def get_next_frame_num(self):
        return self.next_frame_num


def construct_all_detection_info(
    ego_config: "CameraConfig",
    ego_trajectory: "list[trajectory_3d]",
    all_detections: "list[obj_detection]"
):
    all_detection_info: "List[DetectionInfo]" = []
    if len(all_detections) == 0:
        return all_detection_info

    ego_road_polygon_info = get_largest_polygon_containing_point(ego_config)
    detections_polygon_mapping = get_detection_polygon_mapping(all_detections, ego_config)
    # assert len(all_detections) == len(detections_polygon_mapping)
    for detection in all_detections:
        detection_id, car_loc3d, car_loc2d, car_bbox3d, car_bbox2d = detection
        if detection_id in detections_polygon_mapping:
            road_segment_info = detections_polygon_mapping[detection_id]

            detection_info = DetectionInfo(detection_id,
                                           road_segment_info,
                                           car_loc3d,
                                           car_loc2d,
                                           car_bbox3d,
                                           car_bbox2d,
                                           ego_trajectory,
                                           ego_config,
                                           ego_road_polygon_info)
            all_detection_info.append(detection_info)

    return all_detection_info


def generate_sample_plan(
    video: "Video",
    next_frame_num: int,
    all_detection_info: "List[DetectionInfo]",
    ego_views: "List[shape.geometry.Polygon]",
    view_distance: float,
    fps: int = 20,
):
    ### the object detection with higher priority doesn't necessarily get sampled first,
    # it also based on the sample plan
    sample_plan = SamplePlan(video, next_frame_num, all_detection_info, ego_views, fps=fps)
    sample_plan.update_next_sample_frame_num()
    # sample_plan.generate_sample_plan(view_distance)
    return sample_plan
