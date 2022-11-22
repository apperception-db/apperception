""" Goal to map the road segment to the frame segment
    Now only get the segment of type lane and intersection
    except for the segment that contains the ego camera
    
Usage example:
    from optimization_playground.segment_mapping import map_imgsegment_roadsegment
    from apperception.utils import fetch_camera_config
    
    test_config = fetch_camera_config(test_img, database)
    mapping = map_imgsegment_roadsegment(test_config)
"""

from typing import Tuple, List, Set, NamedTuple

import os
import math
import time
import sys

from ..camera_config import CameraConfig
# sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

import cv2
import numpy as np
import numpy.typing as npt
import pandas as pd
from matplotlib import pyplot as plt
from shapely.geometry import Point, Polygon
from plpygis import Geometry
import postgis
import psycopg2
import array
# from pyquaternion import Quaternion
pd.get_option("display.max_columns")

from apperception.database import database
# from apperception.utils import fetch_camera_config
from .utils import line_to_polygon_intersection

data_path = '/home/yongming/workspace/research/apperception/v1.0-mini/'
input_video_dir = os.path.join(data_path, 'sample_videos/')
input_video_name = 'CAM_FRONT_n008-2018-08-27.mp4'
input_date = input_video_name.split('_')[-1][:-4]
test_img = 'samples/CAM_FRONT/n008-2018-08-27-11-48-51-0400__CAM_FRONT__1535385108412412.jpg'

# CAMERA_COLUMNS = [
#     "cameraId",
#     "frameId",
#     "frameNum",
#     "filename",
#     "cameraTranslation",
#     "cameraRotation",
#     "cameraIntrinsic",
#     "egoTranslation",
#     "egoRotation",
#     "timestamp",
#     "cameraHeading",
#     "egoHeading",]
#     # "roadDirection"]
# CAM_CONFIG_QUERY = """SELECT * FROM Cameras 
#                     WHERE filename like 'samples/CAM_FRONT/%{date}%' 
#                     ORDER BY frameNum""" 

# _camera_config = database.execute(CAM_CONFIG_QUERY.format(date=input_date))
# camera_config_df = pd.DataFrame(_camera_config, columns=CAMERA_COLUMNS)
# camera_config_df

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
    {ego_translation}::geometry
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
        {start_segment}::geometry,
        {view_distance}
    ) AND
    segmentpolygon.segmenttypes in (
        ARRAY[\'lane\'],
        ARRAY[\'intersection\'],
        ARRAY[\'laneSection\']
    );"""


Float2 = Tuple[float, float]
Float3 = Tuple[float, float, float]
Float22 = Tuple[Float2, Float2]
Segment = Tuple[str, postgis.polygon.Polygon, str | None, float | None]
AnnotatedSegment = Tuple[str, postgis.polygon.Polygon, str | None, float | None, bool]

class RoadSegmentInfo(NamedTuple):
    """
    segment_id: unique segment id
    segment_polygon: tuple of (x, y) coordinates
    segment_type: road segment type
    contains_ego: whether the segment contains ego camera
    ego_config: ego camfig for the frame we asks info for
    facing_relative: float
    fov_lines: field of view lines
    """
    segment_id: int
    segment_polygon: Polygon
    segment_type: str
    segment_heading: float
    contains_ego: bool
    ego_config: "CameraConfig"
    fov_lines: "Tuple[Float22, Float22]"


# CameraSegmentMapping = namedtuple('cam_segment_mapping', ['cam_segment', 'road_segment_info'])
class CameraSegmentMapping(NamedTuple):
    cam_segment: "List[npt.NDArray[np.floating]]"
    road_segment_info: "RoadSegmentInfo"
    

def road_segment_contains(ego_config: "CameraConfig")\
        -> List[SegmentPolygonWithHeading]:
    query = psycopg2.sql.SQL(SEGMENT_CONTAIN_QUERY).format(
        ego_translation=psycopg2.sql.Literal(postgis.point.Point(*ego_config.ego_translation[:2]))
    )

    return database.execute(query)

def find_segment_dwithin(start_segment: "AnnotatedSegment",
                         view_distance=50) -> "List[SegmentPolygonWithHeading]":
    _, start_segment_polygon, _, _, _ = start_segment
    query = psycopg2.sql.SQL(SEGMENT_DWITHIN_QUERY).format(
        start_segment=psycopg2.sql.Literal(start_segment_polygon),
        view_distance=psycopg2.sql.Literal(view_distance)
    )

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

class HashableAnnotatedSegment:
    val: "AnnotatedSegment"

    def __init__(self, val: "AnnotatedSegment"):
        self.val = val
    
    def __hash__(self):
        h1 = hash(self.val[0])
        h2 = hash(self.val[1].wkt_coords)
        h3 = hash(self.val[2:])
        return hash((h1, h2, h3))

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, HashableAnnotatedSegment):
            return False
        return self.val == __o.val

def construct_search_space(
    ego_config: "CameraConfig",
    view_distance: float = 50.
) -> "List[AnnotatedSegment]":
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

    return [
        s.val
        for s in {
            # To remove duplicates
            *map(HashableAnnotatedSegment, all_contain_segment),
            *map(HashableAnnotatedSegment, segment_within_distance)
        }
    ]

def get_fov_lines(ego_config: "CameraConfig", ego_fov: float = 70.) -> Tuple[Float22, Float22]:
    '''
    return: two lines representing fov in world coord
            ((lx1, ly1), (lx2, ly2)), ((rx1, ry1), (rx2, ry2))
    '''

    # TODO: accuracy improvement: find fov in 3d -> project down to z=0 plane
    ego_heading = ego_config.ego_heading
    x_ego, y_ego = ego_config.ego_translation[:2]
    left_degree = math.radians(ego_heading + ego_fov/2 + 90)
    left_fov_line = ((x_ego, y_ego), 
        (x_ego + math.cos(left_degree)*50, 
         y_ego + math.sin(left_degree)*50))
    right_degree = math.radians(ego_heading - ego_fov/2 + 90)
    right_fov_line = ((x_ego, y_ego), 
        (x_ego + math.cos(right_degree)*50, 
         y_ego + math.sin(right_degree)*50))
    return left_fov_line, right_fov_line

def intersection(fov_line: Tuple[Float22, Float22], segmentpolygon: Polygon):
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
    ego_translation: "Float3",
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


def worl2pixel_factory(config: "CameraConfig"):
    def world2pixel(point3d: "Float2") -> "npt.NDArray[np.floating]":
        point = np.copy((*point3d, 0))

        point -= config.camera_translation
        point = np.dot(config.camera_rotation.inverse.rotation_matrix, point)

        view = np.array(config.camera_intrinsic)
        viewpad = np.eye(4)
        viewpad[: view.shape[0], : view.shape[1]] = view

        point = point.reshape((3, 1))
        point = np.concatenate((point, np.ones((1, 1))))
        point = np.dot(viewpad, point)
        point = point[:3, :]

        point = point / point[2:3, :].repeat(3, 0).reshape(3, 1)
        return point[:2, :]
    return world2pixel


def construct_mapping(
    decoded_road_segment: "List[Float2]",
    frame_size: Tuple[int, int],
    fov_lines: Tuple[Float22, Float22],
    segmentid: str,
    segmenttype: str,
    segmentheading: float,
    contains_ego: bool,
    ego_config: "CameraConfig"
) -> "CameraSegmentMapping | None":
    """
    Given current road segment
    determine whether add it to the mapping
     - segment that contains the ego
     - segment that is larger than 100 pixel x pixel
    """
    ego_translation = ego_config.ego_translation[:2]

    deduced_cam_segment = list(map(worl2pixel_factory(ego_config), decoded_road_segment))
    assert len(deduced_cam_segment) == len(decoded_road_segment)
    if contains_ego:
        keep_cam_segment_point = deduced_cam_segment
        keep_road_segment_point = decoded_road_segment
    else:
        keep_cam_segment_point: "List[npt.NDArray[np.floating]]" = []
        keep_road_segment_point: "List[Float2]" = []
        for current_cam_point, current_road_point in zip(deduced_cam_segment, decoded_road_segment):
            if in_frame(current_cam_point, frame_size) and \
                in_view(current_road_point, ego_translation, fov_lines):
                keep_cam_segment_point.append(current_cam_point)
                keep_road_segment_point.append(current_road_point)
    if contains_ego or (len(keep_cam_segment_point) > 2 
        and Polygon(tuple(keep_cam_segment_point)).area > 100):
        return CameraSegmentMapping(
            keep_cam_segment_point, 
            RoadSegmentInfo(
                segmentid,
                Polygon(keep_road_segment_point),
                segmenttype,
                segmentheading,
                contains_ego,
                ego_config,
                fov_lines
            )
        )

def map_imgsegment_roadsegment(
    ego_config: "CameraConfig",
    frame_size: "Tuple[int, int]" = (1600, 900)
) -> List[CameraSegmentMapping]:
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
        return not in_view(point, ego_config.ego_translation, fov_lines)

    for road_segment in search_space:
        segmentid, segmentpolygon, segmenttype, segmentheading, contains_ego = road_segment
        XYs: "Tuple[array.array[float], array.array[float]]" = Geometry(segmentpolygon.to_ewkb()).exterior.shapely.xy
        assert isinstance(XYs, tuple)
        assert isinstance(XYs[0], array.array), type(XYs[0])
        assert isinstance(XYs[1], array.array), type(XYs[1])
        assert isinstance(XYs[0][0], float), type(XYs[0][0])
        assert isinstance(XYs[1][0], float), type(XYs[1][0])
        segmentpolygon_points = list(zip(*XYs))
        segmentpolygon = Polygon(segmentpolygon_points)
        decoded_road_segment = segmentpolygon_points
        if not contains_ego:
            road_filter = all(map(not_in_view, segmentpolygon_points))
            if road_filter:
                continue

            intersection_points = intersection(fov_lines, segmentpolygon)
            decoded_road_segment += intersection_points

        current_mapping = construct_mapping(
            decoded_road_segment, frame_size, fov_lines, segmentid,
            segmenttype, segmentheading, contains_ego, ego_config)
        if current_mapping is not None:
            mapping.append(current_mapping)

    print('total mapping time: ', time.time() - start_time)
    return mapping

# def visualization(test_img_path: str, test_config: Dict[str, Any], mapping: Tuple):
#     """
#     visualize the mapping from camera segment to road segment
#     for testing only
#     """
#     from moviepy.editor import VideoClip
#     from moviepy.video.io.bindings import mplfig_to_npimage
#     frame = cv2.imread(test_img_path)
#     fig, axs = plt.subplots()
#     axs.set_aspect('equal', 'datalim')
#     x_ego, y_ego = test_config['egoTranslation'][:2]
#     axs.plot(x_ego, y_ego, color='green', marker='o', markersize=5)
#     colormap = plt.cm.get_cmap('hsv', len(mapping))
#     i = 0
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     display_video = cv2.VideoWriter('in_videw_test_display.avi',fourcc, 1, (1600, 900))
#     for cam_segment, road_segment_info in mapping:
#         color = colormap(i)
#         xs = [point[0] for point in road_segment_info.segment_polygon.exterior.coords]
#         ys = [point[1] for point in road_segment_info.segment_polygon.exterior.coords]
#         segmenttype = road_segment_info.segment_type
#         axs.fill(xs, ys, alpha=0.5, fc=color, ec='none')
#         axs.text(np.mean(np.array(xs)), np.mean(np.array(ys)), 
#                 segmenttype if segmenttype else '')
#         current_plt = mplfig_to_npimage(fig)
#         i += 1
        
#         fov_lines = road_segment_info.fov_lines
#         axs.plot([p[0] for p in fov_lines[0]], [p[1] for p in fov_lines[0]], color='red', marker='o', markersize=2)
#         axs.plot([p[0] for p in fov_lines[1]], [p[1] for p in fov_lines[1]], color='red', marker='o', markersize=2)

#         display_frame = frame.copy()
#         cv2.polylines(display_frame, [np.array(cam_segment, np.int32).reshape((-1, 1, 2))], True, (0, 255, 0), 2)
#         display_frame[:current_plt.shape[0], :current_plt.shape[1]] = current_plt
#         display_video.write(display_frame)

#     display_video.release()

# if __name__ == '__main__':
#     test_img_path = os.path.join(data_path, test_img)
#     test_config = fetch_camera_config(
#         test_img, 
#         database)
#     mapping = map_imgsegment_roadsegment(test_config)
#     visualization(test_img_path, test_config, mapping)