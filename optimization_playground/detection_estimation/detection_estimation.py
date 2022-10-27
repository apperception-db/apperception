import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir)))

from segment_mapping import *
from utils import *
from sample_plan_algorithm import SAMPLE_ALGORITHMS

"""
Detection Info:
    frame_segment: frame segment that the detection lies in
    road_segment_info: segment_mapping roadSegmentInfo
    car_loc(3d)
    car_bbox(3d)
    car_bbox(2d)
"""

class detectionInfo:
    def __init__(self,
                 car_id,
                 frame_segment,
                 road_segment_info,
                 car_loc3d,
                 car_loc2d,
                 car_bbox3d,
                 car_bbox2d,
                 ego_trajectory,
                 ego_config):
        self.car_id = car_id
        self.frame_segment = frame_segment
        self.road_segment_info = road_segment_info
        self.car_loc3d = car_loc3d
        self.car_loc2d = car_loc2d
        self.car_bbox3d = car_bbox3d
        self.car_bbox2d = car_bbox2d
        self.ego_trajectory = ego_trajectory
        self.ego_config = ego_config
        self.compute_geo_info()

    def compute_geo_info(self):
        self.distance = compute_distance(self.car_loc3d, self.ego_config.ego_loc)
        self.segment_area = compute_area(self.car_bbox2d)

#TODO
class samplePlan:
    def __init__():
        self.next_action = None
        self.heuristic = None
        self.action = None
        self.current_priority = None

    def add(priority, sample_action, time_threshold = 0.5):
        if current_priority is None:
            current_priority = priority
            action = sample_action
        else:
            if sample_action.estimated < self.action.estimated:
                if (priority >= current_priority or 
                    sample_action.estimated_time/self.action.estimated_time < \
                        time_threshold):
                    current_priority = priority
                    action = sample_action

    def get_next_frame_num():
        if self.action is None:
            return 0
        return self.action.next_frame_num

def generate_single_sample_action(detection_info, view_distance, fps):
    """Generate a sample plan for the given detection of a single car

    Condition 1: detected car is driving towards ego, opposite direction
    Condition 2: detected car driving along the same direction as ego
    Condition 3: detected car and ego are driving into each other
                 i.e. one to north, one from west to it
    Condition 4: detected car and ego are driving away from each other
                 i.e. one to north, one from it to east
    Return: a list of actions
    """
    relative_direction = relative_direction_to_ego(detection_info, ego_config)
    sample_action_alg = get_sample_action_alg(relative_direction)
    return sample_action_alg(detection_info, view_distance, fps)

def yolo_detect(current_frame):
    #TODO
    pass

def construct_all_detection_info(current_frame, cam_segment_mapping, ego_trajectory):
    all_detection_info = []
    for detection in yolo_detect(current_frame):
        car_loc3d, car_loc2d, car_bbox3d, car_bbox2d = detection
        related_mapping = detection_to_img_segment(car_loc2d, cam_segment_mapping)
        if related_mapping is None:
            cam_segment, road_segment_info = None, None
        cam_segment, road_segment_info = related_mapping
        detection_info = detectionInfo(detection.id,
                                       cam_segment,
                                       road_segment_info,
                                       car_loc3d,
                                       car_loc2d,
                                       car_bbox3d,
                                       car_bbox2d,
                                       ego_trajectory,
                                       ego_config)
        all_detection_info.append(detection_info)
    return all_detection_info

def generate_sample_plan(all_detection_info, ego_trajectory, ego_config, view_distance, fps):
    compute_priority(all_detection_info)
    ### the object detection with higher priority doesn't necessarily get sampled first,
    # it also based on the sample plan
    sample_plan = samplePlan()
    for detection_info in all_detection_info:
        sample_plan.add(
            detection_info.priority,
            generate_single_sample_action(detection_info, ego_trajectory, view_distance, fps))
    return sample_plan

def detection_estimation(sorted_ego_config, video, view_distance, fps):
    """Estimated detection throughout the whole video
    
    Return: metadata of the video including all object trajectories
            and other useful info tbd
             
    """
    ego_trajectory = get_ego_trajectory(video, sorted_ego_config)
    i = 0
    next_frame_num = 0
    current_frame = video.capture()
    next_sample_plan = samplePlan()
    while current_frame:
        next_frame_num = next_sample_plan.get_next_frame_num()
        if i == next_frame_num:
            current_ego_config = sorted_ego_config[i]
            cam_segment_mapping = map_imgsegment_roadsegment(current_ego_config)
            all_detection_info = construct_all_detection_info(current_frame, cam_segment_mapping)
            next_sample_plan = generate_sample_plan(all_detection_info, ego_trajectory, current_ego_config)
            next_frame_num = sample_plan.get_next_frame_num()
        i += 1
