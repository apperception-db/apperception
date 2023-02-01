""" Goal to map the road segment to the frame segment
    Now only get the segment of type lane and intersection
    except for the segment that contains the ego camera

Usage example:
    from optimization_playground.segment_mapping import map_imgsegment_roadsegment
    from apperception.utils import fetch_camera_config

    test_config = fetch_camera_config(test_img, database)
    mapping = map_imgsegment_roadsegment(test_config)
"""

from apperception.database import database

import array
import logging
import math
import numpy as np
import numpy.typing as npt
import os
import plpygis
import postgis
import psycopg2
import psycopg2.sql as sql
import shapely
import shapely.geometry
import shapely.wkb
import time
from typing import NamedTuple, Tuple

from ...camera_config import CameraConfig
from .detection_estimation import obj_detection
from .utils import Float2, Float3, Float22, line_to_polygon_intersection
from ...types import DetectionId, obj_detection

logger = logging.getLogger(__name__)

data_path = '/home/yongming/workspace/research/apperception/v1.0-mini/'
input_video_dir = os.path.join(data_path, 'sample_videos/')
input_video_name = 'CAM_FRONT_n008-2018-08-27.mp4'
input_date = input_video_name.split('_')[-1][:-4]
test_img = 'samples/CAM_FRONT/n008-2018-08-01-15-52-19-0400__CAM_FRONT__1533153253912404.jpg'

MAX_POLYGON_CONTAIN_QUERY = psycopg2.sql.SQL("""
WITH max_contain AS (
SELECT
    MAX(ST_Area(p.elementpolygon)) max_segment_area
FROM segmentpolygon AS p
    LEFT OUTER JOIN segment
        ON p.elementid = segment.elementid
WHERE ST_Contains(
    p.elementpolygon,
    {ego_translation}::geometry
)
)
SELECT
    p.elementid,
    p.elementpolygon,
    p.segmenttypes,
    ARRAY_AGG(s.segmentline)::geometry[],
    ARRAY_AGG(s.heading)::real[],
    COUNT(DISTINCT p.elementpolygon),
    COUNT(DISTINCT p.segmenttypes)
FROM max_contain, segmentpolygon AS p
    LEFT OUTER JOIN segment AS s USING (elementid)
WHERE ST_Area(p.elementpolygon) = max_contain.max_segment_area
GROUP BY p.elementid;
""")


class RoadSegmentWithHeading(NamedTuple):
    id: 'str'
    polygon: 'postgis.Polygon'
    road_types: 'list[str]'
    segmentline: 'list[shapely.geometry.LineString]'
    heading: 'list[float]'


class Segment(NamedTuple):
    id: 'str'
    polygon: 'postgis.Polygon'
    road_type: 'str'
    segmentline: 'list[shapely.geometry.LineString]'
    heading: 'list[float]'


class AnnotatedSegment(NamedTuple):
    id: 'str'
    polygon: 'postgis.Polygon'
    road_type: 'str'
    segmentline: 'list[shapely.geometry.LineString]'
    heading: 'list[float]'
    contain: 'bool'


class SegmentLine(NamedTuple):
    line: "shapely.geometry.LineString"
    heading: "float"


class RoadPolygonInfo(NamedTuple):
    """
    id: unique polygon id
    polygon: tuple of (x, y) coordinates
    segment_line: list of tuple of (x, y) coordinates
    segment_type: road segment type
    segment_headings: list of floats
    contains_ego: whether the segment contains ego camera
    ego_config: ego camfig for the frame we asks info for
    fov_lines: field of view lines
    """
    id: str
    polygon: "shapely.geometry.Polygon"
    segment_lines: "list[shapely.geometry.LineString]"
    road_type: str
    segment_headings: "list[float]"
    contains_ego: bool
    ego_config: "CameraConfig"
    fov_lines: "Tuple[Float22, Float22]"


class CameraPolygonMapping(NamedTuple):
    cam_segment: "list[npt.NDArray[np.floating]]"
    road_polygon_info: "RoadPolygonInfo"


def hex_str_to_linestring(hex: 'str'):
    return shapely.geometry.LineString(shapely.wkb.loads(bytes.fromhex(hex)))


def make_road_polygon_with_heading(row: "tuple"):
    eid, polygon, types, lines, headings, *_ = row
    return RoadSegmentWithHeading(
        eid,
        polygon,
        types,
        [*map(hex_str_to_linestring, lines[1:-1].split(':'))],
        headings,
    )


def reformat_return_polygon(segments: "list[RoadSegmentWithHeading]") -> "list[Segment]":
    def _(x: "RoadSegmentWithHeading") -> "Segment":
        i, polygon, types, lines, headings = x
        return Segment(
            i,
            polygon,
            # TODO: fix this hack: all the useful types are 'lane', 'lanegroup',
            # 'intersection' which are always at the last position
            types[-1],
            lines,
            [*map(math.degrees, headings)],
        )
    return list(map(_, segments))


def get_fov_lines(ego_config: "CameraConfig", ego_fov: float = 70.) -> "Tuple[Float22, Float22]":
    '''
    return: two lines representing fov in world coord
            ((lx1, ly1), (lx2, ly2)), ((rx1, ry1), (rx2, ry2))
    '''

    # TODO: accuracy improvement: find fov in 3d -> project down to z=0 plane
    ego_heading = ego_config.ego_heading
    x_ego, y_ego = ego_config.ego_translation[:2]
    left_degree = math.radians(ego_heading + ego_fov / 2 + 90)
    left_fov_line = ((x_ego, y_ego),
                     (x_ego + math.cos(left_degree) * 50,
                      y_ego + math.sin(left_degree) * 50))
    right_degree = math.radians(ego_heading - ego_fov / 2 + 90)
    right_fov_line = ((x_ego, y_ego),
                      (x_ego + math.cos(right_degree) * 50,
                       y_ego + math.sin(right_degree) * 50))
    return left_fov_line, right_fov_line


def intersection(fov_line: Tuple[Float22, Float22], segmentpolygon: "shapely.geometry.Polygon"):
    '''
    return: intersection point: tuple[tuple]
    '''
    left_fov_line, right_fov_line = fov_line
    left_intersection = line_to_polygon_intersection(segmentpolygon, left_fov_line)
    right_intersection = line_to_polygon_intersection(segmentpolygon, right_fov_line)
    return left_intersection + right_intersection


def in_frame(transformed_point: "npt.NDArray", frame_size: Tuple[int, int]):
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


def world2pixel_factory(config: "CameraConfig"):
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


def world2pixel_all(points3d: "list[Float2]", config: "CameraConfig"):
    n = len(points3d)
    points = np.concatenate([np.array(points3d), np.zeros((n, 1))], axis=1)
    assert points.shape == (n, 3)
    points -= config.camera_translation
    assert points.shape == (n, 3)
    points = config.camera_rotation.inverse.rotation_matrix @ points.T
    assert points.shape == (3, n)

    intrinsic = np.array(config.camera_intrinsic)
    assert intrinsic.shape == (3, 3), intrinsic.shape

    points = intrinsic @ points
    assert points.shape == (3, n)

    points = points / points[2:3, :]
    return points.T[:, :2]


def get_largest_polygon_containing_point(
    ego_config: "CameraConfig"
) -> "list[RoadSegmentWithHeading]":
    point = postgis.Point(*ego_config.ego_translation[:2])
    query = MAX_POLYGON_CONTAIN_QUERY.format(
        ego_translation=psycopg2.sql.Literal(point)
    )

    results = database.execute(query)
    if len(results)  > 1:
        for result in results:
            segmenttypes = result[2]
            if 'roadsection' not in segmenttypes:
                results = [result]
                break
    assert len(results) == 1
    result = results[0]
    assert result[5:] == (1, 1)

    output = make_road_polygon_with_heading(result)

    types, line, heading = output[2:5]
    assert line is not None
    assert types is not None
    assert heading is not None
    road_polygon = reformat_return_polygon([output])[0]
    polygonid, roadpolygon, roadtype, segmentlines, segmentheadings = road_polygon
    fov_lines = get_fov_lines(ego_config)
    return RoadPolygonInfo(
        polygonid,
        plpygis.Geometry(roadpolygon.to_ewkb()).shapely,
        segmentlines,
        roadtype,
        segmentheadings,
        True,
        ego_config,
        fov_lines
    )


def map_detections_to_segments(detections: "list"):
    tokens = [*map(lambda x: x.detection_id.obj_order, detections)]
    txs = [*map(lambda x: x.car_loc3d[0], detections)]
    tys = [*map(lambda x: x.car_loc3d[1], detections)]
    tzs = [*map(lambda x: x.car_loc3d[2], detections)]

    _point = sql.SQL("UNNEST({fields}) AS _point (token, tx, ty, tz)").format(
        fields=sql.SQL(',').join(map(sql.Literal, [tokens, txs, tys, tzs]))
    )

    out = sql.SQL("""
    WITH
    Point AS (SELECT * FROM {_point}),
    MaxPolygon AS (
        SELECT token, MAX(ST_Area(Polygon.elementPolygon)) as size
        FROM Point AS p
        JOIN SegmentPolygon AS Polygon
            ON ST_Contains(Polygon.elementPolygon, ST_Point(p.tx, p.ty))
            AND ARRAY ['intersection', 'lane', 'lanegroup', 'lanesection'] && Polygon.segmenttypes
        GROUP BY token
    ),
    MaxPolygonId AS (
        SELECT token, MIN(elementId) as elementId
        FROM Point AS p
        JOIN MaxPolygon USING (token)
        JOIN SegmentPolygon as Polygon
            ON ST_Contains(Polygon.elementPolygon, ST_Point(p.tx, p.ty))
            AND ST_Area(Polygon.elementPolygon) = MaxPolygon.size
            AND ARRAY ['intersection', 'lane', 'lanegroup', 'lanesection'] && Polygon.segmenttypes
        GROUP BY token
    ),
    PointPolygonSegment AS (
        SELECT
            *,
            ST_Distance(ST_Point(tx, ty), ST_MakeLine(startPoint, endPoint)) AS distance
        FROM Point AS p
        JOIN MaxPolygonId USING (token)
        JOIN SegmentPolygon USING (elementId)
        JOIN Segment USING (elementId)
    ),
    MinDis as (
        SELECT token, MIN(distance) as mindistance
        FROM PointPolygonSegment
        GROUP BY token
    )
    SELECT
        p.token,
        p.elementid,
        p.elementpolygon,
        p.segmenttypes,
        ARRAY_AGG(s.segmentline)::geometry[],
        ARRAY_AGG(s.heading)::real[],
        COUNT(DISTINCT p.elementpolygon),
        COUNT(DISTINCT p.segmenttypes)
    FROM PointPolygonSegment AS p
        LEFT OUTER JOIN segment AS s USING (elementid)
    JOIN MinDis USING (token)
    WHERE p.distance = MinDis.mindistance
        AND 'roadsection' != ALL(p.segmenttypes)
    GROUP BY p.elementid, p.token, p.elementpolygon, p.segmenttypes;
    """).format(_point=_point)

    result = database.execute(out)
    return result


def get_detection_polygon_mapping(detections: "list[obj_detection]", ego_config: "CameraConfig"):
    """
    Given a list of detections, return a list of RoadSegmentWithHeading
    """
    start_time = time.time()
    results = map_detections_to_segments(detections)
    assert all(r[6:] == (1, 1) for r in results)
    order_ids, mapped_polygons = [r[0] for r in results], [r[1:] for r in results]
    mapped_polygons = [*map(make_road_polygon_with_heading, mapped_polygons)]
    for row in mapped_polygons:
        types, line, heading = row[2:5]
        assert line is not None
        assert types is not None
        assert heading is not None
    mapped_polygons = reformat_return_polygon(mapped_polygons)
    mapped_road_polygon_info = {}
    fov_lines = get_fov_lines(ego_config)

    def not_in_view(point: "Float2"):
        return not in_view(point, ego_config.ego_translation, fov_lines)

    for order_id, road_polygon in list(zip(order_ids, mapped_polygons)):
        polygonid, roadpolygon, roadtype, segmentlines, segmentheadings = road_polygon
        assert segmentlines is not None
        assert segmentheadings is not None

        assert all(isinstance(l, shapely.geometry.LineString) for l in segmentlines)

        p = plpygis.Geometry(roadpolygon.to_ewkb())
        assert isinstance(p, plpygis.Polygon)
        XYs: "Tuple[array.array[float], array.array[float]]" = p.exterior.shapely.xy
        assert isinstance(XYs, tuple)
        assert isinstance(XYs[0], array.array), type(XYs[0])
        assert isinstance(XYs[1], array.array), type(XYs[1])
        assert isinstance(XYs[0][0], float), type(XYs[0][0])
        assert isinstance(XYs[1][0], float), type(XYs[1][0])
        polygon_points = list(zip(*XYs))
        roadpolygon = shapely.geometry.Polygon(polygon_points)

        decoded_road_polygon_points = polygon_points
        if all(map(not_in_view, polygon_points)):
            continue

        intersection_points = intersection(fov_lines, roadpolygon)
        decoded_road_polygon_points += intersection_points
        keep_road_polygon_points: "list[Float2]" = []
        for current_road_point in decoded_road_polygon_points:
            if in_view(current_road_point, ego_config.ego_translation, fov_lines):
                keep_road_polygon_points.append(current_road_point)
        frame_idx = detections[0].detection_id.frame_idx
        if (len(keep_road_polygon_points) > 2
                        and shapely.geometry.Polygon(tuple(keep_road_polygon_points)).area > 100):
            mapped_road_polygon_info[
                DetectionId(frame_idx=frame_idx, obj_order=order_id)
            ] = RoadPolygonInfo(
                polygonid,
                shapely.geometry.Polygon(keep_road_polygon_points),
                segmentlines,
                roadtype,
                segmentheadings,
                False,
                ego_config,
                fov_lines
            )

    logger.info(f'total mapping time: {time.time() - start_time}')
    return mapped_road_polygon_info
