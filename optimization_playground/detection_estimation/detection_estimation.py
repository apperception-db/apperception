import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir)))

from apperception.database import database
from apperception.utils import F, transformation, fetch_camera_config, fetch_camera_trajectory
from shapely.geometry import Point, Polygon, LineString

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
                 frame_segment,
                 road_segment_info,
                 car_loc3d,
                 car_loc2d,
                 car_bbox3d,
                 car_bbox2d,
                 ego_trajectory,
                 ego_config):
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
    sample_action_alg = SAMPLE_ALGORITHMS[relative_direction]
    return sample_action_alg(detection_info, view_distance, fps)

def generate_sample_plan(all_detection_info, ego_trajectory, ego_config, view_distance, fps):
    compute_priority(all_detection_info)
    ### the object detection with higher priority doesn't necessarily get sampled first,
    # it also based on the sample plan
    sample_plan = samplePlan()
    for detection_info in all_detection_info:
        sample_plan.add(
            detection_info.priority,
            generate_single_sample_action(detection_info, ego_trajectory, view_distance, fps))

def detection_estimation(sorted_ego_config, video):
    """
        Estimate_detection, return all objects info
        i = 0
        while i < len(sorted_ego_config):
            current_ego_config = sorted_ego_config[i]
            cam_segment_mapping = map_imgsegment_roadsegment(current_ego_config)
            all_detection_info = detection_all(decode video frame 1)
            sample_plan = generate_sample_plan(all_detection_info, current_ego_config)
            skipped_framenum = sample_plan.next_framenum - current_ego_config.frame_num
            i += skipped_framenum
             
    """
    pass
