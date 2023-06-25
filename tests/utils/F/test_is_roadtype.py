import pytest
from common import *


@pytest.mark.parametrize("fn, sql", [
    (is_roadtype('intersection'), 'SegmentPolygon.__RoadType__intersection__'),
    (is_roadtype('lane'), 'SegmentPolygon.__RoadType__lane__'),
])
def test_is_roadtype(fn, sql):
    assert gen(fn) == sql
