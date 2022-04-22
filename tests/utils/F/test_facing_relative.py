import pytest
from apperception.utils import fn_to_sql, F


@pytest.mark.parametrize("fn, sql", [
    (lambda o, c: F.facing_relative(o, c.ego, c.timestamp), 
        "facingRelative(T.itemHeadings, C.egoHeading, C.timestamp)"),
    (lambda o, c: F.facing_relative(c.ego, o, c.timestamp), 
        "facingRelative(C.egoHeading, T.itemHeadings, C.timestamp)")
])

@pytest.mark.parametrize("fn, sql", [
    # (lambda o1, o2, c: F.facing_relative(o1, o2, c.timestamp), 
    #     "facingRelative(T.itemHeadings, T2.itemHeadings, C.timestamp)"),
])

def test_facing_relative(fn, sql):
    assert fn_to_sql(fn, ["T", "C"]) == sql


