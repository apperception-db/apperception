from utils import *

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
SAMPLE_ALGORITHMS = {
    SAME_DIRECTION: same_direction_sample_plan,
    OPPOSITE_DIRECTION: opposite_direction_sample_plan,
}

class Action:
    def __init__(self, start_time, finish_action_time, start_loc, end_loc, action_type):
        """Assume the action is always straight line"""
        self.start_time = start_time
        self.finish_time = finish_time
        self.start_loc = loc
        self.end_loc = end_loc
        self.action_type = action_type

def ego_exit_current_segment(ego_trajectory, ego_config):
    current_segment_info = ego_config.road_segment_info
    current_time = ego_config.timestamp
    ego_loc = ego_config.ego_loc
    exit_time, exit_point = time_to_exit_current_segment(
        current_segment_info, current_time, ego_loc, ego_trajectory)
    exit_action = Action(current_time, exit_time, ego_loc, exit_point, action_type=EGO_EXIT_SEGMENT)
    return exit_action

def car_exit_current_segment(detection_info):
    """
        Assumption: detected car drives at max speed
    """
    current_segment_info = detection_info.road_segment_info
    current_time = detection_info.timestamp
    car_loc = detection_info.car_loc
    exit_time, exit_point = time_to_exit_current_segment(current_segment_info, current_time, car_loc)
    exit_action = Action(current_time, exit_time, start_loc=car_loc,
                         end_loc=exit_point, action_type=CAR_EXIT_SEGMENT)
    return exit_action

def car_meet_up_with_ego(detection_info, ego_trajectory, ego_config):
    current_time = ego_config.timestamp
    car2_loc = detection_info.car_loc
    car1_heading = ego_config.heading
    car2_heading = detection_info.road_segment_info.heading
    road_type = detection_info.road_type
    car1_trajectory = ego_trajectory
    ego_loc = ego_config.ego_loc
    meet_up_time, meetup_point = meetup(ego_loc, car2_loc, car1_heading,
                                        car2_heading, road_type, car1_trajectory)
    meet_up_action = Action(current_time, meet_up_time, start_loc=car2_loc,
                            end_loc=meetup_point, action_type=MEET_UP)
    return meet_up_action

def car_exit_view(detection_info, ego_trajectory, ego_config, view_distance):
    current_time = detection_info.timestamp
    ego_loc = ego_config.ego_loc
    car_loc = detection_info.car_loc
    car_heading = detection_info.road_segment_info.heading
    exit_view_time, exit_view_point = time_to_exit_view(
        ego_loc, car_loc, car_heading, ego_trajectory, view_distance)
    exit_view_action = Action(current_time, exit_view_time, start_loc=car_loc,
                              end_loc=exit_view_point, action_type=EXIT_VIEW)

def ego_by_pass_car(detection_info):
    pass

def combine_sample_actions(sample_plan):
    return min(sample_plan, key=lambda x: x.finish_time)

def same_direction_sample_action(detection_info, view_distance):
    ego_trajectory = detection_info.ego_trajectory
    ego_config = detection_info.ego_config
    ego_exit_segment_action = egp_exit_current_segment(ego_trajectory, ego_config)
    car_exit_segment_action = car_exit_current_segment(detection_info)
    car_go_beyong_view_action = car_go_beyond_view(
        detection_info, ego_trajectory, ego_config, view_distance)
    # ego_by_pass_car_action = ego_by_pass_car(detection_info, ego_trajectory, ego_config)
    return combine_sample_actions([ego_exit_segment_action,
                                   car_exit_segment_action,
                                   car_go_beyong_view_action,])
                                   # ego_by_pass_car_action])

def opposite_direction_sample_action(detection_info, view_distance):
    ego_trajectory = detection_info.ego_trajectory
    ego_config = detection_info.ego_config
    ego_exit_segment_action = egp_exit_current_segment(ego_trajectory, ego_config)
    car_exit_segment_action = car_exit_current_segment(detection_info)
    meet_ego_action = meet_up_with_ego(detection_info, ego_trajectory, ego_loc, ego_config)
    return combine_sample_actions([ego_exit_segment_action,
                                   car_exit_segment_action,
                                   meet_ego_action])
