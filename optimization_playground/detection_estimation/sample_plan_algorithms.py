from .utils import *

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
OBJ_BASED_ACTION = [CAR_EXIT_SEGMENT, EXIT_VIEW, MEET_UP]

class Action:
    def __init__(self, start_time, finish_time, start_loc, end_loc, action_type, target_obj_id=None):
        """Assume the action is always straight line"""
        self.start_time = start_time
        self.finish_time = finish_time
        self.invalid_action = False
        if self.finish_time < self.start_time:
            self.invalid_action = True
        self.start_loc = start_loc
        self.end_loc = end_loc
        self.action_type = action_type
        self.estimated_time = self.finish_time - self.start_time
        if action_type and action_type in OBJ_BASED_ACTION:
            assert target_obj_id
            self.target_obj_id = target_obj_id

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f'''action type: {self.action_type},
        start time: {self.start_time},
        finish time: {self.finish_time},
        start loc: {self.start_loc},
        end loc: {self.end_loc}
        estimated time: {self.estimated_time}'''

def ego_exit_current_segment(detection_info, ego_trajectory, ego_config):
    current_segment_info = detection_info.road_segment_info
    current_time = detection_info.timestamp
    ego_loc = ego_config['egoTranslation']
    exit_time, exit_point = time_to_exit_current_segment(
        current_segment_info, current_time, ego_loc, ego_trajectory)
    exit_action = Action(current_time, exit_time, ego_loc, exit_point,
                         action_type=EGO_EXIT_SEGMENT)
    return exit_action

def car_exit_current_segment(detection_info):
    """
        Assumption: detected car drives at max speed
    """
    current_segment_info = detection_info.road_segment_info
    current_time = detection_info.timestamp
    car_loc = detection_info.car_loc3d
    exit_time, exit_point = time_to_exit_current_segment(current_segment_info, current_time, car_loc)
    exit_action = Action(current_time, exit_time, start_loc=car_loc,
                         end_loc=exit_point, action_type=CAR_EXIT_SEGMENT,
                         target_obj_id=detection_info.obj_id)
    return exit_action

def car_meet_up_with_ego(detection_info, ego_trajectory, ego_config):
    current_time = detection_info.timestamp
    car2_loc = detection_info.car_loc3d
    car1_heading = ego_config['egoHeading']
    car2_heading = detection_info.road_segment_info.segment_heading
    road_type = detection_info.road_type
    car1_trajectory = ego_trajectory
    ego_loc = tuple(ego_config['egoTranslation'])
    meet_up_time, meetup_point = meetup(ego_loc, car2_loc, car1_heading,
                                        car2_heading, road_type, current_time, car1_trajectory)
    if meet_up_time < current_time:
        return None
    meet_up_action = Action(current_time, meet_up_time, start_loc=car2_loc,
                            end_loc=meetup_point, action_type=MEET_UP,
                            target_obj_id=detection_info.obj_id)
    return meet_up_action

def car_exit_view(detection_info, ego_trajectory, ego_config, view_distance):
    current_time = detection_info.timestamp
    road_type = detection_info.road_type
    ego_loc = ego_config['egoTranslation']
    car_loc = detection_info.car_loc3d
    car_heading = detection_info.road_segment_info.segment_heading
    exit_view_point, exit_view_time= time_to_exit_view(
        ego_loc, car_loc, car_heading, ego_trajectory, current_time, road_type, view_distance)
    exit_view_action = Action(current_time, exit_view_time, start_loc=car_loc,
                              end_loc=exit_view_point, action_type=EXIT_VIEW,
                              target_obj_id=detection_info.obj_id)
    return exit_view_action

def ego_by_pass_car(detection_info):
    pass

def combine_sample_actions(sample_plan):
    return min(sample_plan, key=lambda x: x.finish_time)

def same_direction_sample_action(detection_info, view_distance):
    ego_trajectory = detection_info.ego_trajectory
    ego_config = detection_info.ego_config
    ego_exit_segment_action = ego_exit_current_segment(detection_info, ego_trajectory, ego_config)
    # print('ego_exit_segment_action', ego_exit_segment_action)
    car_exit_segment_action = car_exit_current_segment(detection_info)
    # print('car_exit_segment_action', car_exit_segment_action)
    car_go_beyong_view_action = car_exit_view(
        detection_info, ego_trajectory, ego_config, view_distance)
    # print('car_go_beyong_view_action', car_go_beyong_view_action)
    # ego_by_pass_car_action = ego_by_pass_car(detection_info, ego_trajectory, ego_config)
    return combine_sample_actions([car_exit_segment_action,
                                   car_go_beyong_view_action,])
                                   # ego_by_pass_car_action])

def opposite_direction_sample_action(detection_info, view_distance):
    ego_trajectory = detection_info.ego_trajectory
    ego_config = detection_info.ego_config
    ego_exit_segment_action = ego_exit_current_segment(detection_info, ego_trajectory, ego_config)
    # print('ego_exit_segment_action', ego_exit_segment_action)
    car_exit_segment_action = car_exit_current_segment(detection_info)
    # print('car_exit_segment_action', car_exit_segment_action)
    meet_ego_action = car_meet_up_with_ego(detection_info, ego_trajectory, ego_config)
    # print('meet_ego_action', meet_ego_action)
    # return car_exit_segment_action
    return combine_sample_actions([car_exit_segment_action,
                                   meet_ego_action])

def get_sample_action_alg(relative_direction):
    if relative_direction == SAME_DIRECTION:
        return same_direction_sample_action
    elif relative_direction == OPPOSITE_DIRECTION:
        return opposite_direction_sample_action
