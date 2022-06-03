import pytest
from apperception.utils import fn_to_sql, F


@pytest.mark.parametrize("fn, sql", [
    (lambda o, c: F.like(o.traj, c.timestamp), 
        "(T.trajCentroids LIKE C.timestamp)"),
    (lambda o, c: F.like(o.object_type, 'human%'), 
        "(T.objectType LIKE 'human%')"),
])

def test_get_x_y(fn, sql):
    assert fn_to_sql(fn, ["T", "C"]) == sql
