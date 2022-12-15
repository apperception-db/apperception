import datetime
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Literal, Union

from .utils import (OPPOSITE_DIRECTION, SAME_DIRECTION, Float2, Float3,
                    ego_departure, meetup, time_to_exit_current_segment,
                    time_to_exit_view, trajectory_3d)

if TYPE_CHECKING:
    from ...camera_config import CameraConfig
    from .detection_estimation import DetectionInfo


logger = logging.getLogger(__name__)


"""
Action:
    trajectory so far
    next timestamp to sample
    next frame to sample and it's frame num
    Heuristic to sample
"""

EGO_EXIT_SEGMENT = 'ego_exit_segment'
CAR_EXIT_SEGMENT = 'car_exit_segment'
EXIT_VIEW = 'exit_view'
MEET_UP = 'meet_up'
EGO_STOP = 'ego_stop'
OBJ_BASED_ACTION = [CAR_EXIT_SEGMENT, EXIT_VIEW, MEET_UP]

ActionType = Literal['ego_exit_segment', 'car_exit_segment', 'exit_view', 'meet_up', 'ego_stop']


@dataclass
class Action:
    start_time: "datetime.datetime"
    finish_time: "datetime.datetime"
    start_loc: "Float2 | Float3"  # TODO: should either be Float2 or Float3
    end_loc: "Float2 | Float3"  # TODO: should either be Float2 or Float3
    action_type: "ActionType"
    target_obj_id: "str | None" = None
    invalid_action: bool = field(init=False)
    estimated_time: "datetime.timedelta" = field(init=False)

    def __post_init__(self):
        self.invalid_action = self.finish_time < self.start_time
        self.estimated_time = self.finish_time - self.start_time
        if self.action_type and self.action_type in OBJ_BASED_ACTION:
            assert self.target_obj_id is not None

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f'''action type: {self.action_type},
        start time: {self.start_time},
        finish time: {self.finish_time},
        start loc: {self.start_loc},
        end loc: {self.end_loc}
        estimated time: {self.estimated_time}'''


def ego_stop(ego_trajectory: "List[trajectory_3d]", ego_config: "CameraConfig"):
    current_time = ego_config.timestamp
    ego_loc = ego_config.ego_translation[:2]
    _ego_stop, ego_departure_time, ego_departure_loc = ego_departure(ego_trajectory, current_time)
    action = None
    if _ego_stop:
        action = Action(current_time, ego_departure_time, ego_loc, ego_departure_loc, action_type=EGO_STOP)
    return _ego_stop, action


def ego_exit_current_segment(
    detection_info: "DetectionInfo",
    ego_trajectory: "List[trajectory_3d]",
    ego_config: "CameraConfig"
):
    current_time = detection_info.timestamp
    ego_loc = ego_config.ego_translation[:2]
    exit_time, exit_point = time_to_exit_current_segment(
        detection_info, current_time, ego_loc, ego_trajectory)
    exit_action = Action(current_time, exit_time, ego_loc, exit_point,
                         action_type=EGO_EXIT_SEGMENT)
    return exit_action


def car_exit_current_segment(detection_info: "DetectionInfo"):
    """
        Assumption: detected car drives at max speed
    """
    current_time = detection_info.timestamp
    car_loc = detection_info.car_loc3d
    exit_time, exit_point = time_to_exit_current_segment(detection_info, current_time, car_loc)
    exit_action = Action(current_time, exit_time, start_loc=car_loc,
                         end_loc=exit_point, action_type=CAR_EXIT_SEGMENT,
                         target_obj_id=detection_info.obj_id)
    return exit_action


def car_meet_up_with_ego(
    detection_info: "DetectionInfo",
    ego_trajectory: "List[trajectory_3d]",
    ego_config: "CameraConfig"
):
    current_time = detection_info.timestamp
    car2_loc = detection_info.car_loc3d
    car1_heading = ego_config.ego_heading
    car2_heading = detection_info.segment_heading
    if car2_heading is None:
        return None
    road_type = detection_info.road_type
    car1_trajectory = ego_trajectory
    ego_loc = tuple(ego_config.ego_translation)
    meet_up_time, meetup_point = meetup(ego_loc, car2_loc, car1_heading,
                                        car2_heading, road_type, current_time, car1_trajectory)
    if meet_up_time < current_time:
        return None
    meet_up_action = Action(current_time, meet_up_time, start_loc=car2_loc,
                            end_loc=meetup_point, action_type=MEET_UP,
                            target_obj_id=detection_info.obj_id)
    return meet_up_action


def car_exit_view(
    detection_info: "DetectionInfo",
    ego_trajectory: "List[trajectory_3d]",
    ego_config: "CameraConfig",
    view_distance: float
):
    current_time = detection_info.timestamp
    road_type = detection_info.road_type
    ego_loc = ego_config.ego_translation
    car_loc = detection_info.car_loc3d
    car_heading = detection_info.segment_heading
    if car_heading is None:
        return None
    exit_view_point, exit_view_time = time_to_exit_view(
        ego_loc, car_loc, car_heading, ego_trajectory, current_time, road_type, view_distance)
    exit_view_action = Action(current_time, exit_view_time, start_loc=car_loc,
                              end_loc=exit_view_point, action_type=EXIT_VIEW,
                              target_obj_id=detection_info.obj_id)
    return exit_view_action


def ego_by_pass_car(detection_info: "DetectionInfo") -> "Action":
    raise Exception()


def combine_sample_actions(sample_plan: "List[Action]"):
    return min(sample_plan, key=lambda x: x.finish_time)


def same_direction_sample_action(detection_info: "DetectionInfo", view_distance: float):
    ego_trajectory = detection_info.ego_trajectory
    ego_config = detection_info.ego_config
    _ego_stop, ego_stop_action = ego_stop(ego_trajectory, ego_config)
    if _ego_stop:
        return ego_stop_action
    ego_exit_segment_action = ego_exit_current_segment(detection_info, ego_trajectory, ego_config)
    # logger.info(f'ego_exit_segment_action {ego_exit_segment_action}')
    car_exit_segment_action = car_exit_current_segment(detection_info)
    # logger.info(f'car_exit_segment_action {car_exit_segment_action}')
    car_go_beyong_view_action = car_exit_view(
        detection_info, ego_trajectory, ego_config, view_distance)
    # logger.info(f'car_go_beyong_view_action {car_go_beyong_view_action}')
    # ego_by_pass_car_action = ego_by_pass_car(detection_info, ego_trajectory, ego_config)
    return combine_sample_actions([ego_exit_segment_action,
                                   car_exit_segment_action,
                                   car_go_beyong_view_action, ])
    # ego_by_pass_car_action])


def opposite_direction_sample_action(detection_info: "DetectionInfo", view_distance: float):
    ego_trajectory = detection_info.ego_trajectory
    ego_config = detection_info.ego_config
    _ego_stop, ego_stop_action = ego_stop(ego_trajectory, ego_config)
    if _ego_stop:
        return ego_stop_action
    ego_exit_segment_action = ego_exit_current_segment(detection_info, ego_trajectory, ego_config)
    # logger.info(f'ego_exit_segment_action {ego_exit_segment_action}')
    car_exit_segment_action = car_exit_current_segment(detection_info)
    # logger.info(f'car_exit_segment_action {car_exit_segment_action}')
    meet_ego_action = car_meet_up_with_ego(detection_info, ego_trajectory, ego_config)
    # logger.info(f'meet_ego_action {meet_ego_action}')
    # return car_exit_segment_action
    actions = [ego_exit_segment_action, car_exit_segment_action]
    if meet_ego_action is not None:
        actions.append(meet_ego_action)
    return combine_sample_actions(actions)


def get_sample_action_alg(relative_direction: "Union[None, Literal['same_direction', 'opposite_direction']]"):
    if relative_direction == SAME_DIRECTION:
        return same_direction_sample_action
    elif relative_direction == OPPOSITE_DIRECTION:
        return opposite_direction_sample_action
