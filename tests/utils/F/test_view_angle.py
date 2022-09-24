import pytest
from common import *


@pytest.mark.parametrize("fn, sql", [
    (view_angle(o.trans@c.time, c.camAbs), 
        "viewAngle(valueAtTimestamp(t0.translations,timestamp),cameraHeading,cameraTranslation)"),
    (view_angle(o.trans@c.time, o.trans@c.time), 
        "viewAngle(valueAtTimestamp(t0.translations,timestamp),(valueAtTimestamp(t0.itemHeadings,timestamp))::real,valueAtTimestamp(t0.translations,timestamp))")
])

def test_view_angle(fn, sql):
    assert gen(fn) == sql


