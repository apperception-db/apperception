import pytest
from apperception.utils import fn_to_sql, F


@pytest.mark.parametrize("fn, sql", [
    (lambda o, c: F.angle_between(o.traj, c.timestamp, o.timestamp), 
        "angleBetween(T.trajCentroids, C.timestamp, T.timestamp)"),
])

def test_angle_between(fn, sql):
    assert fn_to_sql(fn, ["T", "C"]) == sql
