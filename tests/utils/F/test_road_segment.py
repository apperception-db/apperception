import pytest
from apperception.utils import fn_to_sql, F


@pytest.mark.parametrize("fn, sql", [
    (lambda o, c: F.road_segment('road'), 
        "roadSegment('road')")
])
def test_road_direction(fn, sql):
    assert fn_to_sql(fn, ["T", "C"]) == sql


@pytest.mark.parametrize("fn, msg", [
    (lambda o: F.road_segment('not_valid'), 
        "Unsupported road type: not_valid"),
    (lambda o: F.road_segment([]), 
        "Unsupported road type: not_valid"),
])
def test_exception(fn, msg):
    with pytest.raises(Exception) as e_info:
        fn_to_sql(fn, ["T"])
    str(e_info.value) == msg
