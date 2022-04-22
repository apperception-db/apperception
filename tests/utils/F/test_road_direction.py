import pytest
from apperception.utils import fn_to_sql, F


@pytest.mark.parametrize("fn, sql", [
    (lambda o, c: F.road_direction(o, c.timestamp), 
        "roadDirection(T.trajCentroids, C.timestamp)"),
    (lambda o, c: F.road_direction(c.ego, c.timestamp), 
        "roadDirection(C.egoTranslation, C.timestamp)")
])

def test_road_direction(fn, sql):
    assert fn_to_sql(fn, ["T", "C"]) == sql


