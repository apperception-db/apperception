""" Goal of Streaming Detection Optimization:

Input: a video, camera config, road network
Output: estimation of object trajectory
"""

from collections import namedtuple
import os
import math

import cv2
import numpy as np
import pandas as pd
pd.get_option("display.max_columns")

from shapely.geometry import Point as shapely_Point
from plpygis import Geometry
from sympy import Point, Polygon, Line

os.chdir("../")
from apperception.database import database
from apperception.world import empty_world
from apperception.utils import F, transformation

input_video_dir = '/home/yongming/workspace/research/apperception/v1.0-mini/sample_videos/'
input_video_name = 'CAM_FRONT_n008-2018-08-27.mp4'
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

SEGMENT_CONTAIN_QUERY = """select * from segmentpolygon 
                           where ST_Contains(elementpolygon, \'{ego_translation}\'::geometry);"""
SEGMENT_DWITHIN_QUERY = """select * from segmentpolygon
                           where ST_DWithin(elementpolygon, \'{start_segment}\'::geometry, {view_distance});"""

cam_segment_mapping = namedtuple('cam_segment_mapping', ['cam_segment', 'road_segment_info'])

class roadSegmentInfo:
    def __init__(self, segment_id, segment_polygon, segment_type, contains_ego, ego_config):
        self.segment_id = segment_id
        self.segment_polygon = segment_polygon
        self.segment_type = segment_type
        self.contains_ego = contains_ego
        self.ego_config = ego_config
        self.facing_relative = self.facing_relative(ego_config['egoHeading'], segment_id)
    
    def facing_relative(self, ego_heading, segment_id):
        return

def road_segment_contains(ego_config):
    query = SEGMENT_CONTAIN_QUERY.format(
        ego_translation=shapely_Point(*ego_config['egoTranslation']).wkb_hex)

    return database._execute_query(query)

def find_segment_dwithin(start_segment, view_distance=50):
    start_segment_id, start_segment_polygon, segmenttype, contains_ego = start_segment
    query = SEGMENT_DWITHIN_QUERY.format(
        start_segment=start_segment_polygon, view_distance=view_distance)

    return database._execute_query(query)

def reformat_return_segment(segments):
    return list(map(lambda x: (x[0], x[1], tuple(x[2]) if x[2] is not None else None), segments))

def annotate_contain(segments, contain=False):
    for i in range(len(segments)):
        segments[i] = segments[i] + (contain,)

def construct_search_space(ego_config, view_distance=50):
    '''
    road segment: (elementid, elementpolygon, segmenttype, contains_ego?)
    view_distance: in meters, default 50 because scenic standard
    return: set(road_segment)
    '''
    search_space = set()
    all_contain_segment = reformat_return_segment(road_segment_contains(ego_config))
    annotate_contain(all_contain_segment, contain=True)
    search_space.update(all_contain_segment)
    start_segment = all_contain_segment[0]
    
    segment_within_distance = reformat_return_segment(find_segment_dwithin(start_segment))
    annotate_contain(segment_within_distance, contain=False)
    search_space.update(segment_within_distance)
    return search_space

def get_fov_lines(ego_config, ego_fov=70):
    '''
    return: two lines representing fov in world coord
    '''
    ego_heading = ego_config['egoHeading']
    x_ego, y_ego = ego_config['egoTranslation'][:2]
    left_fov_line = ((x_ego, y_ego), 
        (x_ego + math.cos(ego_heading + ego_fov/2 + 90), 
         y_ego + math.sin(ego_heading + ego_fov/2 + 90)))
    right_fov_line = ((x_ego, y_ego), 
        (x_ego + math.cos(ego_heading - ego_fov/2 + 90), 
         y_ego + math.sin(ego_heading - ego_fov/2 + 90)))
    return left_fov_line, right_fov_line

def intersection(fov_line, segmentpolygon):
    '''
    return: intersection point: tuple[tuple]
    '''
    left_fov_line, right_fov_line = fov_line
    left_intersection = segmentpolygon.intersection(Line(*map(Point, left_fov_line)))
    right_intersection = segmentpolygon.intersection(Line(*map(Point, right_fov_line)))
    return left_intersection + right_intersection

def in_frame(transformed_point, frame_size):
    return (transformed_point[0] < frame_size[0]) and (
        transformed_point[1] < frame_size[1])

def map_imgsegment_roadsegment(ego_config, frame_size=(1600, 900)):
    '''
    FULL ALGORITHM:
    Greedy
    road_segment_info: {world_coord, 
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
    fov_lines = get_fov_lines(ego_config)
    search_space = construct_search_space(ego_config)
    mapping = []
    for road_segment in search_space:
        segmentid, segmentpolygon, segmenttype, contains_ego = road_segment
        segmentpolygon_points = tuple(map(
            Point, tuple(zip(*Geometry(segmentpolygon).exterior.shapely.xy))))
        segmentpolygon = Polygon(*segmentpolygon_points)
        intersection_points = tuple(
            intersection(fov_lines, segmentpolygon))
        deduced_road_segment = segmentpolygon_points+intersection_points
        deduced_cam_segment = tuple(map(
            lambda point: transformation(tuple(point)+(0,), ego_config), deduced_road_segment))
        in_view_index = tuple(map(
            lambda point: in_frame(point, frame_size), deduced_cam_segment))
        if sum(in_view_index) > 2:
            keep_road_segment = tuple([
                deduced_road_segment[i] for i in range(len(deduced_road_segment)) if in_view_index[i]])
            keep_cam_segment = tuple([
                deduced_cam_segment[i].flatten() for i in range(len(deduced_cam_segment)) if in_view_index[i]])
            try:
                mapping.append(
                    cam_segment_mapping(
                        keep_cam_segment, 
                        roadSegmentInfo(segmentid, keep_road_segment, segmenttype, contains_ego, ego_config)))
            except:
                print('error', keep_cam_segment, keep_road_segment)
    return mapping

def visualization(frame, mapping):
    for cam_segment, road_segment_info in mapping.items():
        road_segment = road_segment_info.segment_polygon
        cv2.polylines(frame, np.array([cam_segment], dtype=np.int32), True, (0, 255, 0), 2)
    return frame

### Problems: 1. it shouldn't be the intersection of the line, but the vector
###           2. need to plot the view angle and its intersection with the road segments for further investigation
###           3. need to visualize the camera segments
