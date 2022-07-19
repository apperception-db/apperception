import pytest
from apperception.utils import fn_to_sql, F


@pytest.mark.parametrize("fn, sql", [
    (lambda o, c: F.road_direction(o, c.timestamp, c.ego), 
        "roadDirection(T.trajCentroids, C.timestamp, C.egoHeading)"),
    (lambda o, c: F.road_direction(c.ego, c.timestamp, c.ego), 
        "roadDirection(C.egoTranslation, C.timestamp, C.egoHeading)"),
    (lambda o, c: F.road_direction(c.trans, c.timestamp, c.ego), 
        "roadDirection(C.translations, C.timestamp, C.egoHeading)")
])

def test_road_direction(fn, sql):
    assert fn_to_sql(fn, ["T", "C"]) == sql


