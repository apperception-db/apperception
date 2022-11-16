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

from collections import namedtuple
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir)))

from segment_mapping import *
from .utils import *
from .sample_plan_algorithms import *


obj_detection = namedtuple(
    'obj_detection',
    ['id', 'car_loc3d', 'car_loc2d', 'car_bbox3d', 'car_bbox2d'])


class detectionInfo:
    def __init__(self,
                 obj_id,
                 frame_segment,
                 road_segment_info,
                 car_loc3d,
                 car_loc2d,
                 car_bbox3d,
                 car_bbox2d,
                 ego_trajectory,
                 ego_config,
                 ego_road_segment_info):
        self.obj_id = obj_id
        self.frame_segment = frame_segment
        self.road_segment_info = road_segment_info
        self.car_loc3d = car_loc3d
        self.car_loc2d = car_loc2d
        self.car_bbox3d = car_bbox3dDetection
        self.car_bbox2d = car_bbox2d
        self.ego_trajectory = ego_trajectory
        self.ego_config = ego_config
        self.timestamp = self.ego_config['timestamp']
        self.road_type = self.road_segment_info.segment_type
        self.ego_road_segment_info = ego_road_segment_info
        self.compute_geo_info()
        self.compute_priority()

    def compute_geo_info(self):
        self.distance = compute_distance(self.car_loc3d, self.ego_config['egoTranslation'])
        self.segment_area_2d = compute_area(self.car_bbox2d)
        self.relative_direction = relative_direction_to_ego(self.road_segment_info.segment_heading, self.ego_config['egoHeading'])

    def compute_priority(self):
        self.priority = self.segment_area_2d/self.distance

    def generate_single_sample_action(self, view_distance=50):
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
        if sample_action_alg is None:
            return 0, None
        return self.priority, sample_action_alg(self, view_distance)

#TODO
class samplePlan:
    def __init__(self, video, next_frame_num, all_detection_info=None, metadata=None):
        self.video = video
        self.action = None
        self.next_frame_num = next_frame_num
        self.current_priority = None
        self.all_detection_info = all_detection_info
        self.metadata = metadata

    def generate_sample_plan(self, view_distance=50):
        if self.all_detection_info is not None:
            for detection_info in self.all_detection_info:
                priority, sample_action = detection_info.generate_single_sample_action(view_distance)
                self.add(priority, sample_action)

    def add(self, priority, sample_action, time_threshold = 0.5):
        if sample_action is None or sample_action.invalid_action:
            return
        if self.current_priority is None:
            self.current_priority = priority
            self.action = sample_action
        else:
            if sample_action.estimated_time < self.action.estimated_time:
                if (priority >= self.current_priority or 
                    sample_action.estimated_time/self.action.estimated_time < \
                        time_threshold):
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
        return time_to_nearest_frame(self.video, self.action.finish_time)

    def get_next_frame_num(self, next_frame_num):
        if self.get_next_sample_frame_info():
            next_sample_frame_name, next_sample_frame_num, next_sample_frame_time = self.get_next_sample_frame_info()
            self.next_frame_num = max(next_sample_frame_num, next_frame_num)
        return self.next_frame_num

def yolo_detect(current_frame):
    #TODO: return a list of obj_detection
    # onj_detection : namedtuple('id', 'car_loc3d', 'car_loc2d', 'car_bbox3d',
    #   'car_bbox2d')
    return []

def construct_all_detection_info(current_frame, cam_segment_mapping, ego_trajectory,
                                 ego_config, all_detections=None):
    all_detection_info = []
    if all_detections is None:
        all_detections = yolo_detect(current_frame)
    if len(all_detections) == 0:
        return all_detection_info
    ego_mapping = detection_to_img_segment(
        ego_config['egoTranslation'], cam_segment_mapping, ego=True)
    if ego_mapping is None:
        for mapping in cam_segment_mapping:
            cam_segment, road_segment_info = mapping
        raise ValueError('Ego segment not included')
    ego_cam_segment, ego_road_segment_info = ego_mapping

    for detection in all_detections:
        obj_id, car_loc3d, car_loc2d, car_bbox3d, car_bbox2d = detection
        related_mapping = detection_to_img_segment(car_loc2d, cam_segment_mapping)
        if related_mapping is None:
            continue
        cam_segment, road_segment_info = related_mapping
        
        detection_info = detectionInfo(obj_id,
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

def generate_sample_plan(video, next_frame_num, all_detection_info, view_distance):
    ### the object detection with higher priority doesn't necessarily get sampled first,
    # it also based on the sample plan
    sample_plan = samplePlan(video, next_frame_num, all_detection_info)
    sample_plan.generate_sample_plan(view_distance)
    return sample_plan

def detection_estimation(
    sorted_ego_config: "List[trajectory_3d]", video, start_frame_num: "int", view_distance=50, img_base_dir=''):
    """Estimated detection throughout the whole video

    Args:
        sorted_ego_config: a sorted list of ego_configs of the given video
        video: the video name
        start_frame_num: the frame number to start the sample
        view_distance: the maximum view distance from ego
        img_base_dir: the base directory of the images,
                      TODO:deprecate later
    
    Return: TODO metadata of the video including all object trajectories
            and other useful information
             
    """
    ego_trajectory = get_ego_trajectory(video, sorted_ego_config)
    next_frame_num = start_frame_num
    for i in range(len(sorted_ego_config)-1):
        current_ego_config = sorted_ego_config[i]
        if current_ego_config['frame_num'] != next_frame_num:
            continue
        next_frame_num = sorted_ego_config[i+1]['frame_num']
        cam_segment_mapping = map_imgsegment_roadsegment(current_ego_config)
        current_frame = img_base_dir + current_ego_config['fileName']
        all_detection_info = construct_all_detection_info(
            current_frame, cam_segment_mapping, current_ego_config, ego_trajectory)
        next_sample_plan, next_frame = generate_sample_plan(
            video, current_ego_config, cam_segment_mapping, next_frame_num, all_detection_info=all_detection_info)
        next_frame_num = next_sample_plan.get_next_frame_num(next_frame_num)
