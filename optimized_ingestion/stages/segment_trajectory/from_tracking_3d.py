from typing import Tuple
from apperception.database import database
import psycopg2.sql
import numpy as np

from ...payload import Payload
from ..tracking_3d.from_2d_and_road import From2DAndRoad
from ..tracking_3d import tracking_3d
from . import SegmentTrajectory


class FromTracking3D(SegmentTrajectory):
    def _run(self, payload: "Payload"):

        t3d: "list[tracking_3d.Metadatum] | None" = From2DAndRoad.get(payload)
        assert t3d is not None

        object_mapping: "dict[int, list[tracking_3d.Tracking3DResult]]" = dict()
        for frame in t3d:
            for oid, t in frame.items():
                if oid not in object_mapping:
                    object_mapping[oid] = []
                
                object_mapping[oid].append(t)
        
        points: "list[Tuple[tracking_3d.Tracking3DResult, Tuple[float, float] | None]]" = []
        for traj in object_mapping.values():
            traj.sort(key=lambda t: t.timestamp)

            if len(traj) <= 1:
                points.extend((t, None) for t in traj)
                continue

            points.append((traj[0], _get_direction_2d(traj[0], traj[1])))
            points.append((traj[-1], _get_direction_2d(traj[-2], traj[-1])))

            for prev, curr, next in zip(traj[:-2], traj[1:-1], traj[2:]):
                points.append((curr, _get_direction_2d(prev, next)))
        
        return super()._run(payload)
    pass


def _get_direction_2d(p1: "tracking_3d.Tracking3DResult", p2: "tracking_3d.Tracking3DResult") -> "Tuple[float, float]":
    diff = (p2.point - p1.point)[:2]
    udiff = diff / np.linalg.norm(diff)
    return tuple(udiff)


def map_points_and_directions_to_segment(annotations: "list[Tuple[tracking_3d.Tracking3DResult, Tuple[float, float] | None]]"):
    # tokens = [a.detection_id for a, _ in annotations]
    frame_indices = [a.detection_id.frame_idx for a, _ in annotations]
    object_indices = [a.detection_id.obj_order for a, _ in annotations]
    txs = [a.point[0] for a, _ in annotations]
    tys = [a.point[1] for a, _ in annotations]
    tzs = [a.point[2] for a, _ in annotations]
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
        SELECT token, MIN(ST_Area(Polygon.elementPolygon)) as size
        FROM Point AS p
        JOIN SegmentPolygon AS Polygon
            ON ST_Contains(Polygon.elementPolygon, ST_Point(p.tx, p.ty))
            AND ARRAY ['intersection', 'lane', 'lanegroup', 'lanesection'] && Polygon.segmenttypes
        GROUP BY token
    ),
    MinPolygonId AS (
        SELECT token, MIN(elementId) as elementId
        FROM Point AS p
        JOIN MinPolygon USING (token)
        JOIN SegmentPolygon as Polygon
            ON ST_Contains(Polygon.elementPolygon, ST_Point(p.tx, p.ty))
            AND ST_Area(Polygon.elementPolygon) = MinPolygon.size
            AND ARRAY ['intersection', 'lane', 'lanegroup', 'lanesection'] && Polygon.segmenttypes
        GROUP BY token
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
        JOIN MinPolygonId USING (token)
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
        SELECT token, MIN(distance) as mindistance
        FROM PointPolygonSegment
        GROUP BY token
    ),
    MinDisMinAngle as (
        SELECT token, MIN(LEAST(pps.anglediff, 360-pps.anglediff)) as minangle
        FROM PointPolygonSegment AS pps
        JOIN MinDis USING (token)
        WHERE pps.distance = MinDis.mindistance
        GROUP BY token
    )

    SELECT token, elementid, segmentid
    FROM PointPolygonSegment
    JOIN MinDis USING (token)
    JOIN MinDisMinAngle USING (token)
    WHERE PointPolygonSegment.distance = MinDis.mindistance
        AND PointPolygonSegment.anglediff = MinDisMinAngle.minangle
    """).format(_point=_point)
    
    result = database.execute(out)
    return result
