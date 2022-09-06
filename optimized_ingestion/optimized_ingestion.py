import os
import pandas as pd
import numpy as np
import math

os.chdir("../")
from apperception.database import database
from apperception.world import empty_world
from apperception.utils import F
import sys
sys.path.insert(1, './Yolov5_StrongSORT_OSNet')

from filters import Filter
from typing import Any, Dict, List, Tuple
from frame import Frame
from pipeline import Pipeline

### Constants ###
SAMPLING_RATE = 2
CAMERA_ID = "scene-0757"
TEST_FILE_REG = 'samples/CAM_FRONT/%2018-08-01-15%'
TEST_FILE_DIR = '/home/yongming/workspace/research/apperception/v1.0-mini/'
TEST_TRACK_FILE = "/home/yongming/workspace/research/apperception_new_local/apperception/optimized_ingestion/tracks/CAM_FRONT.txt"
os.remove(TEST_TRACK_FILE) if os.path.exists(TEST_TRACK_FILE) else None

CAMERA_COLUMNS = [
    "cameraId",
    "frameId",
    "frameNum",
    "filename",
    "cameraTranslation",
    "cameraRotation",
    "cameraIntrinsic",
    "egoTranslation",
    "egoRotation",
    "timestamp",
    "cameraHeading",
    "egoHeading",
    "cameraTranslationAbs",
    "roadDirection"]

"""
Helper Functions
TODO: Clean Up
"""
def convert_frame_to_map(frames):
    map_frame = dict(zip(CAMERA_COLUMNS, frames[:12]))
    return map_frame

def transform_to_world(frame_coordinates, ego_translation, ego_rotation):
    ### TODO: get world coordinates
    return frame_coordinates

def get_obj_trajectory(tracking_df, ego_config):
    '''
    returned object info is a dictionary that looks like this:
    {object_id:{frame_idx:[], #need to use the frame idx of the video to get the camera config for each frame
                trajectory:[]}
    '''
    obj_info = {}
    grouped_trajectory = tracking_df.groupby(by=["object_id"])
    for name, group in grouped_trajectory:
        obj_info[name] = {}
        
        object_df = group[[
            'frame_idx', 'object_id', 'object_type', 'bbox_left', 'bbox_top', 'bbox_w', 'bbox_h']]
        object_df = object_df.reset_index(drop=True)
        framenums = group.frame_idx.tolist()
        
        ### get ego_config for each framenum
        transformation_config = ego_config.iloc[framenums]
        transformation_config = transformation_config.reset_index(drop=True)
        
        object_with_ego = pd.concat([object_df, transformation_config], axis=1)
        ### for each coordinate, transform
        obj_trajectory = []
        obj_bboxes = []
        for index, row in object_with_ego.iterrows():
            obj_trajectory.append(transform_to_world(
                                    frame_coordinates=(row['bbox_left']+row['bbox_w']//2, 
                                                       row['bbox_top']+row['bbox_h']//2), 
                                    ego_translation=row['egoTranslation'],
                                    ego_rotation=row['egoRotation']))
            obj_bboxes.append(transform_to_world(
                                frame_coordinates=(row['bbox_left'], row['bbox_top'], 
                                                   row['bbox_left']+row['bbox_w'], 
                                                   row['bbox_top']+row['bbox_h']),
                                ego_translation=row['egoTranslation'],
                                ego_rotation=row['egoRotation']))
        obj_info[name]['frame_idx'] = object_with_ego[['frame_idx']]
        obj_info[name]['trajectory'] = obj_trajectory
        obj_info[name]['bbox'] = obj_bboxes
    return obj_info


def facing_relative(prev_traj_point, next_traj_point, current_ego_heading):
    diff = (next_traj_point[0] - prev_traj_point[0], next_traj_point[1] - prev_traj_point[1])
    diff_heading = math.degrees(np.arctan2(diff[1], diff[0])) - 90
    result = ((diff_heading - current_ego_heading) % 360 + 360) % 360
    return result

def facing_relative_check(obj_info, threshold, ego_config):
    facing_relative_filtered = {}
    for obj_id in obj_info:
        frame_idx = obj_info[obj_id]['frame_idx'].frame_idx
        trajectory = obj_info[obj_id]['trajectory']
        ego_heading = ego_config.iloc[frame_idx.tolist()].egoHeading.tolist()
        filtered_idx = frame_idx[:len(ego_heading)-1][[facing_relative(trajectory[i], trajectory[i+1], ego_heading[i]) > threshold for i in range(len(ego_heading)-1)]]
        facing_relative_filtered[obj_id] = filtered_idx
    return facing_relative_filtered


# Filter used to filter the by how close ego is to an inview segment of some type
class InViewFilter(Filter):
    def __init__(self, distance: float, segment_type: str) -> None:
        self.distance = distance
        self.segment_type = segment_type

    def filter(self, frames: List[Frame], metadata: Dict[Any, Any]) -> Tuple[List[Frame], Dict[Any, Any]]:
        intersection_filtered = []

        # TODO: Connection to DB for each execution might take too much time, do all at same time
        for frame in frames:
            # use sql in order to make use of mobilitydb features. TODO: Find python alternative
            query = f"SELECT TRUE WHERE minDistance('{frame.ego_translation}', '{self.segment_type}') < {self.distance}" 
            result = database._execute_query(query)
            if result:
                intersection_filtered.append(frame)

        return intersection_filtered, metadata

if __name__ == "__main__":
    pipeline = Pipeline() 

    pipeline.add_filter(filter=InViewFilter(distance=10, segment_type="intersection"))