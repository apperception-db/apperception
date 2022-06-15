import pytest
from apperception.utils import fn_to_sql, F


@pytest.mark.parametrize("fn, sql", [
    (lambda o, c: F.contains_all('intersection', [o, 2] @ c.timestamp), 
        f"""(EXISTS(
        SELECT intersection.id
        FROM intersection
            JOIN SegmentPolygon
                ON SegmentPolygon.elementId = intersection.id
            JOIN unnest(ARRAY[valueAtTimestamp(T,C.timestamp),valueAtTimestamp(2,C.timestamp)]) point
                ON ST_Covers(SegmentPolygon.elementPolygon, point)
        GROUP BY intersection.id
        HAVING COUNT(point) = 2
    ))"""),
])

def test_angle_between(fn, sql):
    assert fn_to_sql(fn, ["T", "C"]) == sql
