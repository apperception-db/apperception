import pytest
from apperception.utils import fn_to_sql, F


@pytest.mark.parametrize("fn, sql", [
    (lambda o, c: F.contained_margin(o.traj, c.timestamp, 0.5, o.timestamp), 
        "contained_margin(T.trajCentroids, C.timestamp, 0.5, T.timestamp)"),
])

def test_angle_between(fn, sql):
    print(fn_to_sql(fn, ["T", "C"]))
    assert fn_to_sql(fn, ["T", "C"]) == sql
