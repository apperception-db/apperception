import pytest
from apperception.utils import fn_to_sql, F


@pytest.mark.parametrize("fn, sql", [
    (lambda o, c: F.get_x(o.traj, c.timestamp), 
        "getX(T.trajCentroids, C.timestamp)"),
    (lambda o, c: F.get_x(c.ego, c.timestamp), 
        "getX(C.egoTranslation, C.timestamp)"),
    (lambda o, c: F.get_y(c.ego, c.timestamp), 
        "getY(C.egoTranslation, C.timestamp)"),
    (lambda o, c: F.get_y(o.traj, c.timestamp), 
        "getY(T.trajCentroids, C.timestamp)"),
    (lambda o, c: F.get_z(o.traj, c.timestamp), 
        "getZ(T.trajCentroids, C.timestamp)"),
])

def test_get_x_y(fn, sql):
    assert fn_to_sql(fn, ["T", "C"]) == sql
