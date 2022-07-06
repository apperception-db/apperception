import pytest
from apperception.utils import fn_to_sql, F


@pytest.mark.parametrize("fn, sql", [
    (lambda o, c: F.view_angle(o.trans, c.cameraAbs, c.timestamp), 
        "viewAngle(T.translations, C.cameraHeading, C.cameraTranslationAbs, C.timestamp)")
])

def test_view_angle(fn, sql):
    assert fn_to_sql(fn, ["T", "C"]) == sql


