import pytest
from apperception.utils import fn_to_sql, F


@pytest.mark.parametrize("fn, sql", [
    (lambda o, c: F.convert_camera(o.traj, c, c.timestamp), "ConvertCamera(T.trajCentroids, C.cameraTranslation, C.timestamp)"),
    (lambda o, c: F.convert_camera(o.traj, c.ego, "2004-10-19 10:23:54"), "ConvertCamera(T.trajCentroids, C.egoTranslation, '2004-10-19 10:23:54')"),
    (lambda o, c: F.convert_camera(o, c.ego, c.timestamp), "ConvertCamera(T.trajCentroids, C.egoTranslation, C.timestamp)"),
])
def test_convert_to_camera(fn, sql):
    assert fn_to_sql(fn, ["T", "C"]) == sql


def test_bbox():
    try:
        fn_to_sql(lambda o, c: F.convert_camera(o.bbox, c, c.timestamp), ["T", "C"])
    except Exception as e:
        assert str(e) == "We do not support bbox yet"
