import pytest
from apperception.utils import fn_to_sql, F


@pytest.mark.parametrize("fn, sql", [
    (lambda o, c: F.road_segment('road'), 
        "roadSegment('road')")
])

def test_road_direction(fn, sql):
    assert fn_to_sql(fn, ["T", "C"]) == sql


