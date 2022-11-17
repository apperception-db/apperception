""" Goal to map the road segment to the frame segment
    Now only get the segment of type lane and intersection
    except for the segment that contains the ego camera
    
Usage example:
    from optimization_playground.segment_mapping import map_imgsegment_roadsegment
    from apperception.utils import fetch_camera_config
    
    test_config = fetch_camera_config(test_img, database)
    mapping = map_imgsegment_roadsegment(test_config)
"""

from typing import Any, Dict, Tuple, List, Set, Union

from collections import namedtuple
import os
import math
import time
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from shapely.geometry import Point, Polygon, LineString
from plpygis import Geometry
import postgis
pd.get_option("display.max_columns")

from apperception.database import database
from apperception.utils import F, transformation, fetch_camera_config
from detection_estimation.utils import line_to_polygon_intersection

data_path = '/home/yongming/workspace/research/apperception/v1.0-mini/'
input_video_dir = os.path.join(data_path, 'sample_videos/')
input_video_name = 'CAM_FRONT_n008-2018-08-27.mp4'
input_date = input_video_name.split('_')[-1][:-4]
test_img = 'samples/CAM_FRONT/n008-2018-08-27-11-48-51-0400__CAM_FRONT__1535385108412412.jpg'

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
    "roadDirection"]
CAM_CONFIG_QUERY = """SELECT * FROM Cameras 
                    WHERE filename like 'samples/CAM_FRONT/%{date}%' 
                    ORDER BY frameNum""" 

camera_config = database.execute(CAM_CONFIG_QUERY.format(date=input_date))
camera_config_df = pd.DataFrame(camera_config, columns=CAMERA_COLUMNS)
camera_config_df

SegmentPolygonWithHeading = Tuple[
    str,
    postgis.polygon.Polygon,
    List[str] | None,
    float | None,
]
SEGMENT_CONTAIN_QUERY = """
SELECT
    segmentpolygon.elementid,
    segmentpolygon.elementpolygon,
    segmentpolygon.segmenttypes,
    segment.heading
FROM segmentpolygon 
    LEFT OUTER JOIN segment
        ON segmentpolygon.elementid = segment.elementid
WHERE ST_Contains(
    segmentpolygon.elementpolygon,
    \'{ego_translation}\'::geometry
);
"""

SEGMENT_DWITHIN_QUERY = """
SELECT
    segmentpolygon.elementid,
    segmentpolygon.elementpolygon,
    segmentpolygon.segmenttypes,
    segment.heading
FROM segmentpolygon 
    LEFT OUTER JOIN segment
        ON segmentpolygon.elementid = segment.elementid
WHERE ST_DWithin(
        elementpolygon,
        \'{start_segment}\'::geometry,
        {view_distance}
    ) AND
    segmentpolygon.segmenttypes in (
        ARRAY[\'lane\'],
        ARRAY[\'intersection\'],
        ARRAY[\'laneSection\']
    );"""

cam_segment_mapping = namedtuple('cam_segment_mapping', ['cam_segment', 'road_segment_info'])

Float2 = Tuple[float, float]
Float22 = Tuple[Float2, Float2]
Segment = Tuple[str, postgis.polygon.Polygon, str | None, float | None]
AnnotatedSegment = Tuple[str, postgis.polygon.Polygon, str | None, float | None, bool]

class roadSegmentInfo:
    def __init__(
        self, 
        segment_id: int,
        segment_polygon: Polygon,
        segment_type: str,
        segment_heading: float,
        contains_ego: bool,
        ego_config: Dict[str, Any],
        fov_lines: Tuple[Float22, Float22]):
        """
        segment_id: unique segment id
        segment_polygon: tuple of (x, y) coordinates
        segment_type: road segment type
        contains_ego: whether the segment contains ego camera
        ego_config: ego camfig for the frame we asks info for
        facing_relative: float
        fov_lines: field of view lines
        """
        self.segment_id = segment_id
        self.segment_polygon = segment_polygon
        self.segment_type = segment_type
        self.segment_heading = segment_heading
        self.contains_ego = contains_ego
        self.ego_config = ego_config
        self.fov_lines = fov_lines
    

def road_segment_contains(ego_config: Dict[str, Any])\
        -> List[SegmentPolygonWithHeading]:
    query = SEGMENT_CONTAIN_QUERY.format(
        ego_translation=Point(*ego_config['egoTranslation']).wkb_hex)

    return database.execute(query)

def find_segment_dwithin(start_segment: "AnnotatedSegment",
                         view_distance=50) -> "List[SegmentPolygonWithHeading]":
    _, start_segment_polygon, _, _, _ = start_segment
    query = SEGMENT_DWITHIN_QUERY.format(
        start_segment=start_segment_polygon, view_distance=view_distance)

    return database.execute(query)

def reformat_return_segment(segments: "List[SegmentPolygonWithHeading]") -> "List[Segment]":
    def _(x: "SegmentPolygonWithHeading") -> Segment:
        i, polygon, types, heading = x
        return (
            i,
            polygon,
            types[0] if types is not None else None,
            math.degrees(heading) if heading is not None else None,
        )
    return list(map(_, segments))

def annotate_contain(
    segments: "List[Segment]",
    contain: bool = False
) -> "List[AnnotatedSegment]":
    return [s + (contain,) for s in segments]

def construct_search_space(
    ego_config: Dict[str, Any],
    view_distance=50
) -> "Set[AnnotatedSegment]":
    '''
    road segment: (elementid, elementpolygon, segmenttype, heading, contains_ego?)
    view_distance: in meters, default 50 because scenic standard
    return: set(road_segment)
    '''
    all_contain_segment = reformat_return_segment(road_segment_contains(ego_config))
    all_contain_segment = annotate_contain(all_contain_segment, contain=True)
    start_segment = all_contain_segment[0]
    
    segment_within_distance = reformat_return_segment(find_segment_dwithin(start_segment, view_distance))
    segment_within_distance = annotate_contain(segment_within_distance, contain=False)

    return {
        *all_contain_segment,
        *segment_within_distance
    }

def get_fov_lines(ego_config: Dict[str, Any], ego_fov=70) -> Tuple[Float22, Float22]:
    '''
    return: two lines representing fov in world coord
            ((lx1, ly1), (lx2, ly2)), ((rx1, ry1), (rx2, ry2))
    '''

    # TODO: accuracy improvement: find fov in 3d -> project down to z=0 plane
    ego_heading: float = ego_config['egoHeading']
    x_ego, y_ego: "Float2" = ego_config['egoTranslation'][:2]
    left_degree = math.radians(ego_heading + ego_fov/2 + 90)
    left_fov_line = ((x_ego, y_ego), 
        (x_ego + math.cos(left_degree)*50, 
         y_ego + math.sin(left_degree)*50))
    right_degree = math.radians(ego_heading - ego_fov/2 + 90)
    right_fov_line = ((x_ego, y_ego), 
        (x_ego + math.cos(right_degree)*50, 
         y_ego + math.sin(right_degree)*50))
    return left_fov_line, right_fov_line

def intersection(fov_line: Tuple[Float22, Float22], segmentpolygon: Polygon) -> Tuple[Float2]:
    '''
    return: intersection point: tuple[tuple]
    '''
    left_fov_line, right_fov_line = fov_line
    left_intersection = line_to_polygon_intersection(segmentpolygon, left_fov_line)
    right_intersection = line_to_polygon_intersection(segmentpolygon, right_fov_line)
    return left_intersection + right_intersection

def in_frame(transformed_point: np.array, frame_size: Tuple[int, int]):
    return transformed_point[0] > 0 and transformed_point[0] < frame_size[0] and \
        transformed_point[1] < frame_size[1] and transformed_point[1] > 0

def in_view(
    road_point: "Float2",
    ego_translation: List[float],
    fov_lines: Tuple[Float22, Float22]
) -> bool:
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

def construct_mapping(
    decoded_road_segment: Tuple[Float2],
    frame_size: Tuple[int, int],
    fov_lines: Tuple[Float22, Float22],
    segmentid: str,
    segmenttype: str,
    segmentheading: float,
    contains_ego: bool,
    ego_config: Dict[str, Any]
) -> Tuple[bool, cam_segment_mapping]:
    """
    Given current road segment
    determine whether add it to the mapping
    """
    ego_translation = ego_config['egoTranslation'][:2]
    deduced_cam_segment = tuple(map(
            lambda point: transformation(tuple(point)+(0,), ego_config), decoded_road_segment))
    assert len(deduced_cam_segment) == len(decoded_road_segment)
    if contains_ego:
        keep_cam_segment_point = deduced_cam_segment
        keep_road_segment_point = decoded_road_segment
    else:
        keep_cam_segment_point = []
        keep_road_segment_point = []
        for i in range(len(decoded_road_segment)):
            current_cam_point = deduced_cam_segment[i]
            current_road_point = decoded_road_segment[i]
            if in_frame(current_cam_point, frame_size) and \
                in_view(current_road_point, ego_translation, fov_lines):
                keep_cam_segment_point.append(current_cam_point)
                keep_road_segment_point.append(current_road_point)
    if contains_ego or (len(keep_cam_segment_point) > 2 
        and Polygon(tuple(keep_cam_segment_point)).area > 100):
        return cam_segment_mapping(
            keep_cam_segment_point, 
            roadSegmentInfo(
                segmentid,
                Polygon(keep_road_segment_point),
                segmenttype,
                segmentheading,
                contains_ego,
                ego_config,
                fov_lines
            )
        )

def map_imgsegment_roadsegment(ego_config: Dict[str, Any],
                               frame_size=(1600, 900)) -> List[cam_segment_mapping]:
    """Construct a mapping from frame segment to road segment

    Given an image, we know that different roads/lanes belong to different
    road segment in the road network. We want to find a mapping
    from the road/lane/intersection to the real world road segment so that
    we know which part of the image belong to which part of the real world

    Return List[namedtuple(cam_segment_mapping)]: each tuple looks like this
    (polygon in frame that represents a portion of lane/road/intersection,
     roadSegmentInfo)
    """
    fov_lines = get_fov_lines(ego_config)
    start_time = time.time()
    search_space = construct_search_space(ego_config, view_distance=100)
    mapping = []

    def not_in_view(point: "Float2"):
        return not in_view(point, ego_config['egoTranslation'], fov_lines)

    for road_segment in search_space:
        segmentid, segmentpolygon, segmenttype, segmentheading, contains_ego = road_segment
        XYs: "Tuple[List[float], List[float]]" = Geometry(segmentpolygon).exterior.shapely.xy
        assert isinstance(XYs, tuple)
        assert isinstance(XYs[0], list)
        assert isinstance(XYs[1], list)
        assert isinstance(XYs[0][0], float)
        assert isinstance(XYs[1][0], float)
        segmentpolygon_points = list(zip(*XYs))
        segmentpolygon = Polygon(segmentpolygon_points)
        decoded_road_segment = segmentpolygon_points
        if not contains_ego:
            road_filter = all(map(not_in_view, segmentpolygon_points))
            if road_filter:
                continue

            intersection_points = tuple(
                    intersection(fov_lines, segmentpolygon))
            decoded_road_segment +=intersection_points

        current_mapping = construct_mapping(
            decoded_road_segment, frame_size, fov_lines, segmentid,
            segmenttype, segmentheading, contains_ego, ego_config)
        if current_mapping:
            mapping.append(current_mapping)

    print('total mapping time: ', time.time() - start_time)
    return mapping

def visualization(test_img_path: str, test_config: Dict[str, Any], mapping: Tuple):
    """
    visualize the mapping from camera segment to road segment
    for testing only
    """
    from moviepy.editor import VideoClip
    from moviepy.video.io.bindings import mplfig_to_npimage
    frame = cv2.imread(test_img_path)
    fig, axs = plt.subplots()
    axs.set_aspect('equal', 'datalim')
    x_ego, y_ego = test_config['egoTranslation'][:2]
    axs.plot(x_ego, y_ego, color='green', marker='o', markersize=5)
    colormap = plt.cm.get_cmap('hsv', len(mapping))
    i = 0
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    display_video = cv2.VideoWriter('in_videw_test_display.avi',fourcc, 1, (1600, 900))
    for cam_segment, road_segment_info in mapping:
        color = colormap(i)
        xs = [point[0] for point in road_segment_info.segment_polygon.exterior.coords]
        ys = [point[1] for point in road_segment_info.segment_polygon.exterior.coords]
        segmenttype = road_segment_info.segment_type
        axs.fill(xs, ys, alpha=0.5, fc=color, ec='none')
        axs.text(np.mean(np.array(xs)), np.mean(np.array(ys)), 
                segmenttype if segmenttype else '')
        current_plt = mplfig_to_npimage(fig)
        i += 1
        
        fov_lines = road_segment_info.fov_lines
        axs.plot([p[0] for p in fov_lines[0]], [p[1] for p in fov_lines[0]], color='red', marker='o', markersize=2)
        axs.plot([p[0] for p in fov_lines[1]], [p[1] for p in fov_lines[1]], color='red', marker='o', markersize=2)

        display_frame = frame.copy()
        cv2.polylines(display_frame, [np.array(cam_segment, np.int32).reshape((-1, 1, 2))], True, (0, 255, 0), 2)
        display_frame[:current_plt.shape[0], :current_plt.shape[1]] = current_plt
        display_video.write(display_frame)

    display_video.release()

if __name__ == '__main__':
    test_img_path = os.path.join(data_path, test_img)
    test_config = fetch_camera_config(
        test_img, 
        database)
    mapping = map_imgsegment_roadsegment(test_config)
    visualization(test_img_path, test_config, mapping)