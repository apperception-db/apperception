import datetime
from dataclasses import dataclass
from typing import Tuple

import shapely
import shapely.geometry


@dataclass
class Polygon:
    id: "str"
    polygon: "shapely.geometry.Polygon"
    segments: "list[str] | None"

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, Polygon):
            return False
        
        eq_res = self.polygon.equals(__o.polygon)
        assert isinstance(eq_res, bool)
        return eq_res
    
    def _polygon_coords(self):
        exterior = self.polygon.exterior
        assert exterior is not None
        return tuple(sorted([*exterior.coords][:-1]))

    def __hash__(self) -> int:
        return hash(self._polygon_coords())


@dataclass
class Segment:
    id: "str"
    line: "shapely.geometry.LineString"
    polygon: "Polygon"

    def __post_init__(self):
        assert len(self.line.coords) == 2, self.line
    
    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, Segment):
            return False
        sx, sy = self.line.coords.xy
        ox, oy = __o.line.coords.xy

        if tuple(sx) != tuple(ox) or tuple(sy) != tuple(oy):
            return False
        
        return self.polygon == __o.polygon
    
    def __hash__(self) -> int:
        sx, sy = self.line.coords.xy
        return hash(tuple(sx) + tuple(sy) + self.polygon._polygon_coords())


@dataclass
class SegmentDetection:
    oid: "str"
    did: "str"
    fid: "str"
    timestamp: "datetime.datetime"
    segment: "Segment"
    location: "Tuple[float, float, float]"
    type: "str" = ""
