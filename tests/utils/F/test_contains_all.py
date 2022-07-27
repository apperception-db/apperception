import pytest
from common import *


@pytest.mark.parametrize("fn, sql", [
    (
        contains_all('intersection', [o.trans, 2] @ c.time),
        """(EXISTS(
        SELECT intersection.id
        FROM intersection
            JOIN SegmentPolygon
                ON SegmentPolygon.elementId = intersection.id
            JOIN unnest(ARRAY[valueAtTimestamp(t0.translations,timestamp),valueAtTimestamp(2,timestamp)]) point
                ON ST_Covers(SegmentPolygon.elementPolygon, point)
        GROUP BY intersection.id
        HAVING COUNT(point) = 2
    ))"""
    ),
    (
        contains_all('intersection', c.time),
        """(EXISTS(
        SELECT intersection.id
        FROM intersection
            JOIN SegmentPolygon
                ON SegmentPolygon.elementId = intersection.id
            JOIN unnest(timestamp) point
                ON ST_Covers(SegmentPolygon.elementPolygon, point)
        GROUP BY intersection.id
        HAVING COUNT(point) = cardinality(timestamp)
    ))"""
    ),
])
def test_contain_all(fn, sql):
    assert gen(fn) == sql


@pytest.mark.parametrize("fn, msg", [
    (contains_all(c.time, 1), 
        "Frist argument of contains_all should be a constant, recieved "),
    (contains_all('invalid', 1), 
        "polygon should be either road or lane or lanesection or roadSection or intersection"),
])
def test_exception(fn, msg):
    with pytest.raises(Exception) as e_info:
        gen(fn)
    str(e_info.value) == msg