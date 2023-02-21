from apperception.database import database

import numpy as np
import psycopg2.sql
import postgis
import shapely
import shapely.geometry
import shapely.wkb
from typing import Tuple, NamedTuple

from .construct_segment_trajectory import SegmentPoint
from ...types import DetectionId
from ...payload import Payload
from ..detection_estimation.segment_mapping import RoadPolygonInfo
from ..tracking_3d import tracking_3d
from ..tracking_3d.from_2d_and_road import From2DAndRoad
from . import SegmentTrajectory, SegmentTrajectoryMetadatum


class FromTracking3D(SegmentTrajectory):
    def _run(self, payload: "Payload"):

        t3d: "list[tracking_3d.Metadatum] | None" = From2DAndRoad.get(payload)
        assert t3d is not None

        # Index object trajectories using their object id
        object_map: "dict[int, list[tracking_3d.Tracking3DResult]]" = dict()
        for frame in t3d:
            for oid, t in frame.items():
                if oid not in object_map:
                    object_map[oid] = []

                object_map[oid].append(t)

        # Create a list of detection points with each of its corresponding direction
        points: "list[Tuple[tracking_3d.Tracking3DResult, Tuple[float, float] | None]]" = []
        for traj in object_map.values():
            traj.sort(key=lambda t: t.timestamp)

            if len(traj) <= 1:
                points.extend((t, None) for t in traj)
                continue

            # First and last points' direction
            points.append((traj[0], _get_direction_2d(traj[0], traj[1])))
            points.append((traj[-1], _get_direction_2d(traj[-2], traj[-1])))

            # All other points' direction
            for prev, curr, next in zip(traj[:-2], traj[1:-1], traj[2:]):
                points.append((curr, _get_direction_2d(prev, next)))
        
        # Map a segment to each detection
        # Note: Some detection might be missing due to not having any segment mapped
        segments = map_points_and_directions_to_segment(points)
        
        # Index segments using their detection id
        segment_map: "dict[DetectionId, SegmentMapping]" = {}
        for segment in segments:
            did = DetectionId(*segment[:2])
            assert did not in segment_map
            segment_map[did] = segment


        object_id_to_segmnt_map: "dict[int, list[SegmentPoint]]" = {}
        output: "list[SegmentTrajectoryMetadatum]" = [dict() for _ in t3d]
        for oid, obj in object_map.items():
            segment_trajectory: "list[SegmentPoint]" = []
            object_id_to_segmnt_map[oid] = segment_trajectory

            for det in obj:
                did = det.detection_id
                if did in segment_map:
                    # Detection that can be mapped to a segment
                    segment = segment_map[did]
                    _fid, _oid, polygonid, polygon, segmentid, segmentline, segmentheading = segment
                    assert did.frame_idx == _fid
                    assert did.obj_order == _oid

                    shapely_polygon = shapely.wkb.loads(polygon.to_ewkb(), hex=True)
                    assert isinstance(shapely_polygon, shapely.geometry.Polygon)
                    segment_point = SegmentPoint(
                        did,
                        tuple(det.point.tolist()),
                        det.timestamp,
                        segmentline,
                        segmentheading,
                        # A place-holder for Polygon that only contain polygon id and polygon
                        RoadPolygonInfo(
                            polygonid,
                            shapely_polygon,
                            [],
                            None,
                            [],
                            None,
                            None,
                            None
                        ),
                        oid,
                        None,
                        None,
                    )
                else:
                    # Detection that cannot be mapped to any segment
                    segment_point = SegmentPoint(
                        did,
                        tuple(det.point.tolist()),
                        det.timestamp,
                        None,
                        None,
                        None,
                        oid,
                        None,
                        None
                    )
                
                segment_trajectory.append(segment_point)

                metadatum = output[did.frame_idx]
                assert oid not in metadatum
                metadatum[oid] = segment_point
            
            for prev, next in zip(segment_trajectory[:-1], segment_trajectory[1:]):
                prev.next = next
                next.prev = prev

        return None, {self.classname(): output}


def _get_direction_2d(p1: "tracking_3d.Tracking3DResult", p2: "tracking_3d.Tracking3DResult") -> "Tuple[float, float]":
    diff = (p2.point - p1.point)[:2]
    udiff = diff / np.linalg.norm(diff)
    return tuple(udiff)


class SegmentMapping(NamedTuple):
    fid: "int"
    oid: "int"
    elementid: "str"
    polygon: "postgis.Polygon"
    segmentid: "int"
    line: "postgis.LineString"
    heading: "float"


def map_points_and_directions_to_segment(
    annotations: "list[Tuple[tracking_3d.Tracking3DResult, Tuple[float, float] | None]]"
) -> "list[SegmentMapping]":
    if len(annotations) == 0:
        return []

    frame_indices = [a.detection_id.frame_idx for a, _ in annotations]
    object_indices = [a.detection_id.obj_order for a, _ in annotations]
    txs = [a.point[0] for a, _ in annotations]
    tys = [a.point[1] for a, _ in annotations]
    dxs = [d and d[0] for _, d in annotations]
    dys = [d and d[1] for _, d in annotations]

    _point = psycopg2.sql.SQL("UNNEST({fields}) AS _point (fid, oid, tx, ty, dx, dy)").format(
        fields=psycopg2.sql.SQL(',').join(map(psycopg2.sql.Literal, [frame_indices, object_indices, txs, tys, dxs, dys]))
    )

    out = psycopg2.sql.SQL("""
    DROP FUNCTION IF EXISTS _angle(double precision);
    CREATE OR REPLACE FUNCTION _angle(a double precision) RETURNS double precision AS
    $BODY$
    BEGIN
        RETURN ((a::decimal % 360) + 360) % 360;
    END
    $BODY$
    LANGUAGE 'plpgsql';

    WITH
    Point AS (SELECT * FROM {_point}),
    _SegmentWithDirection AS (
        SELECT
            *,
            ST_X(endPoint) - ST_X(startPoint) AS _x,
            ST_Y(endPoint) - ST_Y(startPoint) AS _y
        FROM Segment
    ),
    SegmentWithDirection AS (
        SELECT
            *,
            (_x / SQRT(POWER(_x, 2) + POWER(_y, 2))) AS dx,
            (_y / SQRT(POWER(_x, 2) + POWER(_y, 2))) AS dy
        FROM _SegmentWithDirection
        WHERE
            _x <> 0 OR _y <> 0
    ),
    MinPolygon AS (
        SELECT fid, oid, MIN(ST_Area(Polygon.elementPolygon)) as size
        FROM Point AS p
        JOIN SegmentPolygon AS Polygon
            ON ST_Contains(Polygon.elementPolygon, ST_Point(p.tx, p.ty))
            AND ARRAY ['intersection', 'lane', 'lanegroup', 'lanesection'] && Polygon.segmenttypes
        GROUP BY fid, oid
    ),
    MinPolygonId AS (
        SELECT fid, oid, MIN(elementId) as elementId
        FROM Point AS p
        JOIN MinPolygon USING (fid, oid)
        JOIN SegmentPolygon as Polygon
            ON ST_Contains(Polygon.elementPolygon, ST_Point(p.tx, p.ty))
            AND ST_Area(Polygon.elementPolygon) = MinPolygon.size
            AND ARRAY ['intersection', 'lane', 'lanegroup', 'lanesection'] && Polygon.segmenttypes
        GROUP BY fid, oid
    ),
    PointPolygonSegment AS (
        SELECT
            *,
            ST_Distance(ST_Point(tx, ty), ST_MakeLine(startPoint, endPoint)) AS distance,
            CASE
                WHEN p.dx IS NULL THEN 0
                WHEN p.dy IS NULL THEN 0
                ELSE _angle(ACOS((p.dx * sd.dx) + (p.dy * sd.dy)) * 180 / PI())
            END AS anglediff
        FROM Point AS p
        JOIN MinPolygonId USING (fid, oid)
        JOIN SegmentPolygon USING (elementId)
        JOIN SegmentWithDirection AS sd USING (elementId)
        WHERE
            'intersection' = Any(SegmentPolygon.segmenttypes)
            OR
            p.dx IS NULL
            OR
            p.dy IS NULL
            OR
            _angle(ACOS((p.dx * sd.dx) + (p.dy * sd.dy)) * 180 / PI()) < 90
            OR
            _angle(ACOS((p.dx * sd.dx) + (p.dy * sd.dy)) * 180 / PI()) > 270
    ),
    MinDis as (
        SELECT fid, oid, MIN(distance) as mindistance
        FROM PointPolygonSegment
        GROUP BY fid, oid
    ),
    MinDisMinAngle as (
        SELECT fid, oid, MIN(LEAST(pps.anglediff, 360-pps.anglediff)) as minangle
        FROM PointPolygonSegment AS pps
        JOIN MinDis USING (fid, oid)
        WHERE pps.distance = MinDis.mindistance
        GROUP BY fid, oid
    )

    SELECT fid, oid, elementid, elementpolygon, segmentid, segmentline, segmentheading
    FROM PointPolygonSegment
    JOIN MinDis USING (fid, oid)
    JOIN MinDisMinAngle USING (fid, oid)
    WHERE PointPolygonSegment.distance = MinDis.mindistance
        AND PointPolygonSegment.anglediff = MinDisMinAngle.minangle
    """).format(_point=_point)

    result = database.execute(out)
    return [*map(SegmentMapping._make, result)]