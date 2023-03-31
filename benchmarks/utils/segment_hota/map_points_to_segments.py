from typing import Any
from os import environ

import psycopg2
import psycopg2.sql as sql
import postgis.psycopg
import postgis.polygon
import postgis.linestring
from pyquaternion import quaternion

from shapely import wkb
import shapely.geometry

from .types import SegmentDetection, Segment, Polygon

connection = psycopg2.connect(
        dbname=environ.get("AP_DB", "mobilitydb"),
        user=environ.get("AP_USER", "docker"),
        host=environ.get("AP_HOST", "localhost"),
        port=environ.get("AP_PORT", "25432"),
        password=environ.get("AP_PASSWORD", "docker"),
    )
connection.autocommit = True
cursor = connection.cursor()
postgis.psycopg.register(connection)
# mobilitydb.psycopg.register(connection)


def execute(
    query: "str | sql.Composable",
    val: "list[Any] | None" = None
):
    cursor.execute(query, val)
    return cursor.fetchall()


QUERY = """
DROP FUNCTION IF EXISTS angle(double precision);
CREATE OR REPLACE FUNCTION angle(a double precision) RETURNS double precision AS
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
        AND Polygon.location = p.location
        AND ARRAY ['intersection', 'lane', 'lanegroup', 'lanesection'] && Polygon.segmenttypes
    GROUP BY token
),
MinPolygonId AS (
    SELECT token, MIN(elementId) as elementId
    FROM Point AS p
    JOIN MinPolygon USING (token)
    JOIN SegmentPolygon as Polygon
        ON ST_Contains(Polygon.elementPolygon, ST_Point(p.tx, p.ty))
        AND Polygon.location = p.location
        AND ST_Area(Polygon.elementPolygon) = MinPolygon.size
        AND ARRAY ['intersection', 'lane', 'lanegroup', 'lanesection'] && Polygon.segmenttypes
    GROUP BY token
),
PointPolygonSegment AS (
    SELECT
        *,
        ST_Distance(ST_Point(tx, ty), ST_MakeLine(startPoint, endPoint)) AS distance,
        angle(ACOS((p.dx * sd.dx) + (p.dy * sd.dy)) * 180 / PI()) AS anglediff
    FROM Point AS p
    JOIN MinPolygonId USING (token)
    JOIN SegmentPolygon USING (elementId)
    JOIN SegmentWithDirection AS sd USING (elementId)
    WHERE
        angle(ACOS((p.dx * sd.dx) + (p.dy * sd.dy)) * 180 / PI()) < 90
        OR
        angle(ACOS((p.dx * sd.dx) + (p.dy * sd.dy)) * 180 / PI()) > 270
        OR
        'intersection' = Any(SegmentPolygon.segmenttypes)
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
),
Out as (
    SELECT token, elementid, elementpolygon, segmentid, segmentline
    FROM PointPolygonSegment
    JOIN MinDis USING (token)
    JOIN MinDisMinAngle USING (token)
    WHERE PointPolygonSegment.distance = MinDis.mindistance
        AND PointPolygonSegment.anglediff = MinDisMinAngle.minangle
)

SELECT * FROM Out
"""


def map_points_to_segments(annotations: "list[dict[str, Any]]") -> "list[SegmentDetection]":
    if len(annotations) == 0:
        return []

    tokens = [*map(lambda x: x['token'], annotations)]
    txs = [*map(lambda x: x['translation'][0], annotations)]
    tys = [*map(lambda x: x['translation'][1], annotations)]
    tzs = [*map(lambda x: x['translation'][2], annotations)]
    ds = [*map(lambda x: quaternion.Quaternion(x['rotation']).rotate([1, 0, 0]), annotations)]
    dxs = [*map(lambda x: x[0], ds)]
    dys = [*map(lambda x: x[1], ds)]
    locations = [*map(lambda x: x['location'], annotations)]
    
    _point = sql.SQL("UNNEST({fields}) AS _point (token, tx, ty, tz, dx, dy, location)").format(
        fields=sql.SQL(',').join(map(sql.Literal, [tokens, txs, tys, tzs, dxs, dys, locations]))
    )
    
    out = sql.SQL(QUERY).format(_point=_point)
    
    result = execute(out)

    print('result length', len(result))


    annotation_map = {
        a['token']: a
        for a
        in annotations
    }

    def tuple_to_segment_detection(
        r: "tuple[str, str, postgis.polygon.Polygon, int, postgis.linestring.LineString]"
    ):
        token, elementid, elementpolygon, segmentid, segmentline = r

        line = wkb.loads(segmentline.to_ewkb(), hex=True)
        assert isinstance(line, shapely.geometry.LineString)

        polygon = wkb.loads(elementpolygon.to_ewkb(), hex=True)
        assert isinstance(polygon, shapely.geometry.Polygon)

        annotation = annotation_map[token]

        return SegmentDetection(
            annotation['instance_token'],
            token,
            annotation['sample_data_token'],
            annotation['timestamp'],
            Segment(
                str(segmentid),
                line,
                Polygon(str(elementid), polygon, None)
            ),
            annotation['translation'],
            annotation['category']
        )

    return [*map(tuple_to_segment_detection, result)]