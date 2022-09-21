"""
Goal of Streaming Detection Optimization:
Input: a video, camera config, road network
Output: estimation of object trajectory
"""

import os
import pandas as pd
import numpy as np
import math

os.chdir("../")
from apperception.database import database
from apperception.world import empty_world
from apperception.utils import F

input_video_dir = '/home/yongming/workspace/research/apperception/v1.0-mini/sample_videos/'
input_video_name = 'CAM_BACK_n008-2018-08-30.mp4'
input_date = input_video_name.split('_')[-1][:-4]

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
CAM_CONFIG_QUERY = """SELECT * FROM Cameras 
                    WHERE filename like 'samples/CAM_FRONT/%{date}%' 
                    ORDER BY frameNum""" 

camera_config = database._execute_query(CAM_CONFIG_QUERY.format(date=input_date))
camera_config_df = pd.DataFrame(camera_config, columns=CAMERA_COLUMNS)
camera_config_df

def map_roadsegment_imgsegment():
    '''
    FULL ALGORITHM:
    Greedy
    road_segment: {world_coord, 
                   segment_type, 
                   ego_in_segment?, 
                   facing_relative(ego_heading, segment_direction)}
    1. Get the lines of fov in world coord
    2. For road_segment in search_space:
        intersect_point = intersection(fov_lines, road_segment)
        cam_segment = filter([road_segment_world_coord, intersection_point], 
                              lambda point: transform(point) in frame)
        if cam_segment is valid:
            append_to_mapping({cam_segments:road_segment})
            
        search_space.append(find_next_segment(current_segment))
    '''

def find_next_segment(ego_loc, ego_heading, current_segment):
    '''
    current_segment: a segment in world coord
    if current_segment is None:
        return world_segment_contains(ego_loc)
    else:
        next_segment = find_closest_segment(current_segment)
        if keep next_segment:
            return get_properties(next_segment)