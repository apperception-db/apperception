""" Goal of Streaming Detection Optimization:

Input: a video, camera config, road network
Output: estimation of object trajectory
"""

from collections import namedtuple
import os
import math
import time

import cv2
import numpy as np
import pandas as pd
pd.get_option("display.max_columns")

from shapely.geometry import Point, Polygon, LineString
from plpygis import Geometry

os.chdir('../')
from apperception.database import database
from apperception.world import empty_world
from apperception.utils import F, transformation, fetch_camera_config

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
    def __init__(self, segment_id, segment_polygon, segment_type, contains_ego, ego_config, fov_lines):
        """
        segment_id: string
        segment_polygon: tuple of sympy.Point
        segment_type: string
        contains_ego: boolean
        ego_config: dict
        facing_relative: float
        fov_lines: tuple(tuple(Point, Point), tuple(Point, Point))
        """
        self.segment_id = segment_id
        self.segment_polygon = segment_polygon
        self.segment_type = segment_type
        self.contains_ego = contains_ego
        self.ego_config = ego_config
        self.facing_relative = self.facing_relative(ego_config['egoHeading'], segment_id)
        self.fov_lines = fov_lines
    
    def facing_relative(self, ego_heading, segment_id):
        return

def road_segment_contains(ego_config):
    query = SEGMENT_CONTAIN_QUERY.format(
        ego_translation=Point(*ego_config['egoTranslation']).wkb_hex)

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
    
    segment_within_distance = reformat_return_segment(find_segment_dwithin(start_segment, view_distance))
    annotate_contain(segment_within_distance, contain=False)
    search_space.update(segment_within_distance)
    return search_space

def get_fov_lines(ego_config, ego_fov=70):
    '''
    return: two lines representing fov in world coord
    '''
    ego_heading = ego_config['egoHeading']
    x_ego, y_ego = ego_config['egoTranslation'][:2]
    left_degree = math.radians(ego_heading + ego_fov/2 + 90)
    left_fov_line = ((x_ego, y_ego), 
        (x_ego + math.cos(left_degree)*50, 
         y_ego + math.sin(left_degree)*50))
    right_degree = math.radians(ego_heading - ego_fov/2 + 90)
    right_fov_line = ((x_ego, y_ego), 
        (x_ego + math.cos(right_degree)*50, 
         y_ego + math.sin(right_degree)*50))
    return left_fov_line, right_fov_line

def intersection(fov_line, segmentpolygon):
    '''
    return: intersection point: tuple[tuple]
    '''
    left_fov_line, right_fov_line = fov_line
    left_intersection = tuple(LineString(left_fov_line).intersection(segmentpolygon).coords)
    right_intersection = tuple(LineString(right_fov_line).intersection(segmentpolygon).coords)
    return left_intersection + right_intersection

def in_frame(transformed_point, frame_size):
    return transformed_point[0] > 0 and transformed_point[0] < frame_size[0] and \
        transformed_point[1] < frame_size[1] and transformed_point[1] > 0

def in_view(road_point, ego_translation, fov_lines):
    '''
    return if the road_point is on the left of the left fov line and 
                                on the right of the right fov line
    '''
    left_fov_line, right_fov_line = fov_lines
    Ax, Ay = ego_translation[:2]
    Mx, My = road_point
    left_fov_line_x, left_fov_line_y = left_fov_line[1]
    right_fov_line_x, right_fov_line_y = right_fov_line[1]
    return (left_fov_line_x - Ax) * (My - Ay) - (left_fov_line_y - Ay) * (Mx - Ax) <= 0 and \
              (right_fov_line_x - Ax) * (My - Ay) - (right_fov_line_y - Ay) * (Mx - Ax) >= 0

def construct_mapping(decoded_road_segment, frame_size, fov_lines, segmentid,
                      segmenttype, contains_ego, ego_config):
    ego_translation = ego_config['egoTranslation'][:2]
    if contains_ego:
        decoded_road_segment += (ego_translation,)
    deduced_cam_segment = tuple(map(
            lambda point: transformation(tuple(point)+(0,), ego_config), decoded_road_segment))
    assert len(deduced_cam_segment) == len(decoded_road_segment)
    keep_cam_segment_point = []
    keep_road_segment_point = []
    for i in range(len(decoded_road_segment)):
        current_cam_point = deduced_cam_segment[i]
        current_road_point = decoded_road_segment[i]
        if in_frame(current_cam_point, frame_size) and \
            in_view(current_road_point, ego_translation, fov_lines):
            keep_cam_segment_point.append(current_cam_point)
            keep_road_segment_point.append(current_road_point)
    return (len(keep_cam_segment_point) > 2, 
            cam_segment_mapping(
                keep_cam_segment_point, 
                roadSegmentInfo(
                    segmentid,
                    keep_road_segment_point,
                    segmenttype,
                    contains_ego,
                    ego_config,
                    fov_lines
                ))
            )

def map_imgsegment_roadsegment(ego_config, frame_size=(1600, 900)):
    '''
    FULL ALGORITHM:
    Greedy
    road_segment_info: {segmentid,
                        segmentpolygon,
                        segment_type, 
                        ego_in_segment?,
                        ego_config,
                        fov_line,
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
    start_time = time.time()
    search_space = construct_search_space(ego_config, view_distance=100)
    print('construct_search_space time: ', time.time() - start_time)
    mapping = []
    total_mapping_time = 0
    total_intersection_compute_time = 0
    total_decode_segment_time = 0
    cardinality = len(search_space)
    for road_segment in search_space:
        start_analyze_time = time.time()
        segmentid, segmentpolygon, segmenttype, contains_ego = road_segment
        segmentpolygon_points = tuple(zip(*Geometry(segmentpolygon).exterior.shapely.xy))
        segmentpolygon = Polygon(segmentpolygon_points)
        decode_segment_time = time.time()
        total_decode_segment_time += decode_segment_time - start_analyze_time

        road_filter = all(map(
            lambda point: not in_view(
                point, ego_config['egoTranslation'], fov_lines), 
            segmentpolygon_points))
        if road_filter:
            continue

        intersection_points = tuple(
            intersection(fov_lines, segmentpolygon))
        decoded_road_segment = segmentpolygon_points+intersection_points
        compute_intersection_time = time.time()
        total_intersection_compute_time += compute_intersection_time - decode_segment_time

        valid_mapping, current_mapping = construct_mapping(
            decoded_road_segment, frame_size, fov_lines, segmentid,
            segmenttype, contains_ego, ego_config)
        if valid_mapping:
            mapping.append(current_mapping)
        total_mapping_time += time.time() - compute_intersection_time
    print('total_decode_segment_time: ', total_decode_segment_time)
    print('total intersection compute time', total_intersection_compute_time)
    print('avg intersection compute time', total_intersection_compute_time/cardinality)
    print('total mapping time: ', total_mapping_time)
    print('avg mapping time: ', total_mapping_time/cardinality)
    print('total time: ', time.time() - start_time)
    return mapping

if __name__ == '__main__':
    test_config = fetch_camera_config(
        'samples/CAM_FRONT/n008-2018-08-27-11-48-51-0400__CAM_FRONT__1535385105912404.jpg', 
        database)
    mapping = map_imgsegment_roadsegment(test_config)