""" Goal to map the road segment to the frame segment
    Now only get the segment of type lane and intersection
    except for the segment that contains the ego camera

Usage example:
    from optimization_playground.segment_mapping import map_imgsegment_roadsegment
    from apperception.utils import fetch_camera_config

    test_config = fetch_camera_config(test_img, database)
    mapping = map_imgsegment_roadsegment(test_config)
"""

import array
import logging
import math
import os
import time
from typing import NamedTuple, Tuple

import numpy as np
import plpygis
import postgis
import psycopg2.sql as sql
import shapely
import shapely.geometry
import shapely.wkb
from scipy.spatial import ConvexHull

from apperception.database import database
from apperception.utils.ingest_road import ROAD_TYPES

from ...camera_config import CameraConfig
from ...types import DetectionId, obj_detection
from .segment_mapping import (
    RoadSegmentWithHeading,
    Segment,
    get_fov_lines,
    in_view,
    make_road_polygon_with_heading,
)
from .utils import Float2, Float22

logger = logging.getLogger(__name__)

data_path = '/home/yongming/workspace/research/apperception/v1.0-mini/'
input_video_dir = os.path.join(data_path, 'sample_videos/')
input_video_name = 'CAM_FRONT_n008-2018-08-27.mp4'
input_date = input_video_name.split('_')[-1][:-4]
test_img = 'samples/CAM_FRONT/n008-2018-08-01-15-52-19-0400__CAM_FRONT__1533153253912404.jpg'

ROAD_TYPE_ARR = list(ROAD_TYPES)

MAX_POLYGON_CONTAIN_QUERY = sql.SQL(f"""
WITH
AvailablePolygon AS (
    SELECT *
    FROM SegmentPolygon as p
    WHERE
        p.location = {{location}}
        AND ST_Contains(
            p.elementpolygon,
            {{ego_translation}}::geometry
        )
),
max_contain AS (
    SELECT MAX(ST_Area(elementpolygon)) max_segment_area
    FROM AvailablePolygon
)
SELECT
    p.elementid,
    p.elementpolygon::geometry,
    ARRAY_AGG(s.segmentline)::geometry[],
    ARRAY_AGG(s.heading)::real[],
    {','.join('p.__RoadType__' + rt + '__' for rt in ROAD_TYPE_ARR)}
FROM max_contain, AvailablePolygon AS p
    LEFT OUTER JOIN segment AS s USING (elementid)
WHERE ST_Area(p.elementpolygon) = max_contain.max_segment_area
GROUP BY p.elementid, p.elementpolygon, {','.join('p.__RoadType__' + rt + '__' for rt in ROAD_TYPE_ARR)};
""")

USEFUL_TYPES = ['lane', 'lanegroup', 'intersection']


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


def reformat_return_polygon(segments: "list[RoadSegmentWithHeading]") -> "list[Segment]":
    def _(x: "RoadSegmentWithHeading") -> "Segment":
        i, polygon, types, lines, headings = x
        type = types[-1]
        for t in types:
            if t in USEFUL_TYPES:
                type = t
                break
        return Segment(
            i,
            polygon,
            type,
            lines,
            [*map(math.degrees, headings)],
        )
    return list(map(_, segments))


def intersection(fov_line: Tuple[Float22, Float22], segmentpolygon: "shapely.geometry.Polygon"):
    '''
    return: intersection point: tuple[tuple]
    '''
    left_fov_line, right_fov_line = fov_line
    # left_intersection = line_to_polygon_intersection(segmentpolygon, left_fov_line)
    # right_intersection = line_to_polygon_intersection(segmentpolygon, right_fov_line)
    # return left_intersection + right_intersection
    return []


def get_largest_polygon_containing_point(ego_config: "CameraConfig"):
    point = postgis.Point(*ego_config.ego_translation[:2])
    query = MAX_POLYGON_CONTAIN_QUERY.format(
        ego_translation=sql.Literal(point),
        location=sql.Literal(ego_config.location)
    )

    results = database.execute(query)
    if len(results) > 1:
        for result in results:
            segmenttypes = result[2]
            if 'roadsection' not in segmenttypes:
                results = [result]
                break
    assert len(results) == 1
    result = results[0]
    assert len(result) == 4 + len(ROAD_TYPE_ARR)

    output = make_road_polygon_with_heading(result)

    types, line, heading = output[2:5]
    assert line is not None
    assert types is not None
    assert heading is not None
    road_polygon = reformat_return_polygon([output])[0]
    polygonid, roadpolygon, roadtype, segmentlines, segmentheadings = road_polygon
    fov_lines = get_fov_lines(ego_config)

    polygon = shapely.wkb.loads(roadpolygon.to_ewkb(), hex=True)
    assert isinstance(polygon, shapely.geometry.Polygon)

    return RoadPolygonInfo(
        polygonid,
        polygon,
        segmentlines,
        roadtype,
        segmentheadings,
        True,
        ego_config,
        fov_lines
    )


def map_detections_to_segments(detections: "list[obj_detection]", ego_config: "CameraConfig"):
    tokens = [*map(lambda x: x.detection_id.obj_order, detections)]
    points = [postgis.Point(d.car_loc3d[0], d.car_loc3d[1]) for d in detections]

    location = ego_config.location

    _points = np.array([[d.car_loc3d[0], d.car_loc3d[1]] for d in detections])
    if len(detections) >= 3:
        ch = ConvexHull(_points)
        convex = _points[[*ch.vertices, ch.vertices[0]]]
    else:
        convex = _points

    out = sql.SQL("""
    WITH
    Point AS (
        SELECT *
        FROM UNNEST(
            {tokens},
            {points}::geometry(Point)[]
        ) AS _point (token, point)
    ),
    AvailablePolygon AS (
        SELECT *
        FROM SegmentPolygon
        WHERE location = {location}
        AND ST_Intersects(SegmentPolygon.elementPolygon, {convex}::geometry(MultiPoint))
        AND (SegmentPolygon.__RoadType__intersection__
        OR SegmentPolygon.__RoadType__lane__
        OR SegmentPolygon.__RoadType__lanegroup__
        OR SegmentPolygon.__RoadType__lanesection__)
    ),
    MaxPolygon AS (
        SELECT token, MAX(ST_Area(Polygon.elementPolygon)) as size
        FROM Point AS p
        JOIN AvailablePolygon AS Polygon
            ON ST_Contains(Polygon.elementPolygon, p.point)
        GROUP BY token
    ),
    MaxPolygonId AS (
        SELECT token, MIN(elementId) as elementId
        FROM Point AS p
        JOIN MaxPolygon USING (token)
        JOIN AvailablePolygon as Polygon
            ON ST_Contains(Polygon.elementPolygon, p.point)
            AND ST_Area(Polygon.elementPolygon) = MaxPolygon.size
        GROUP BY token
    ),
    PointPolygonSegment AS (
        SELECT
            *,
            ST_Distance(p.point, segmentLine) AS distance
        FROM Point AS p
        JOIN MaxPolygonId USING (token)
        JOIN AvailablePolygon USING (elementId)
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
        AND NOT p.__RoadType__roadsection__
    GROUP BY p.elementid, p.token, p.elementpolygon, p.segmenttypes;
    """).format(
        tokens=sql.Literal(tokens),
        points=sql.Literal(points),
        convex=sql.Literal(postgis.MultiPoint(map(tuple, convex))),
        location=sql.Literal(location),
    )

    result = database.execute(out)
    return result


def get_detection_polygon_mapping(detections: "list[obj_detection]", ego_config: "CameraConfig"):
    """
    Given a list of detections, return a list of RoadSegmentWithHeading
    """
    start_time = time.time()
    results = map_detections_to_segments(detections, ego_config)
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
        frame_idx = detections[0].detection_id.frame_idx
        det_id = DetectionId(frame_idx=frame_idx, obj_order=order_id)
        if det_id in mapped_road_polygon_info:
            print("skipped")
            continue
        polygonid, roadpolygon, roadtype, segmentlines, segmentheadings = road_polygon
        assert segmentlines is not None
        assert segmentheadings is not None

        assert all(isinstance(line, shapely.geometry.LineString) for line in segmentlines)

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
        if (len(keep_road_polygon_points) > 2
                and shapely.geometry.Polygon(tuple(keep_road_polygon_points)).area > 40):
            mapped_road_polygon_info[det_id] = RoadPolygonInfo(
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
