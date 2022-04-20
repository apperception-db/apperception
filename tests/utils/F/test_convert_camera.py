import pytest
from apperception.utils import fn_to_sql, F


@pytest.mark.parametrize("fn, sql", [
    (lambda o, c: F.convert_camera(o.traj, c), "ConvertCamera(T.trajCentroids, C.cameraTranslation, C.timestamp)"),
    (lambda o, c: F.convert_camera(o.traj, c.ego), "ConvertCamera(T.trajCentroids, C.egoTranslation, C.timestamp)"),
    (lambda o, c: F.convert_camera(o, c.ego), "ConvertCamera(T.trajCentroids, C.egoTranslation, C.timestamp)"),
])
def test_convert_to_camera(fn, sql):
    assert fn_to_sql(fn, ["T", "C"]) == sql


def test_bbox():
    try:
        fn_to_sql(lambda o, c: F.convert_camera(o.bbox, c), ["T", "C"])
    except Exception as e:
        assert str(e) == "We do not support bbox yet"
