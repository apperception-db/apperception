import pytest
from apperception.utils import fn_to_sql, F


@pytest.mark.parametrize("fn, sql", [
    (lambda o, c: F.ahead(o, c.ego, c.timestamp), 
        "ahead(T.trajCentroids, C.egoTranslation, C.egoHeading, C.timestamp)"),
    (lambda o, c: F.ahead(c.ego, o, c.timestamp), 
        "ahead(C.egoTranslation, T.trajCentroids, T.itemHeadings, C.timestamp)"),
    (lambda o, c: F.ahead(o, o, c.timestamp), 
        "ahead(T.trajCentroids, T.trajCentroids, T.itemHeadings, C.timestamp)"),
])

def test_get_x_y(fn, sql):
    assert fn_to_sql(fn, ["T", "C"]) == sql
