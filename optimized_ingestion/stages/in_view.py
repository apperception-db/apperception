from typing import TYPE_CHECKING, List, Optional, Tuple

from bitarray import bitarray

from apperception.database import database

from .stage import Stage

if TYPE_CHECKING:
    from ..payload import Payload
    from ..camera_config import Float3


class InView(Stage):
    def __init__(self, distance: float, segment_type: str) -> None:
        self.distance = distance
        self.segment_type = segment_type

    def __call__(self, payload: "Payload") -> "Tuple[Optional[bitarray], Optional[list]]":
        keep = bitarray(payload.keep)
        points: "List[Float3]" = []
        indices: "List[int]" = []
        for i, f in enumerate(payload.video):
            if keep[i]:
                points.append(f.ego_translation)
                indices.append(i)

        points_str = ",\n".join(map(_tuple_to_point, points))
        results = database._execute_query(f"""
            SELECT minDistance(p, '{self.segment_type}') < {self.distance}
            FROM UNNEST(ARRAY[{points_str}]) AS points(p)""")

        for i, (r,) in enumerate(results):
            keep[indices[i]] = r

        return keep, None


def _tuple_to_point(t: "Float3"):
    return f"'POINT Z ({' '.join(map(str, t))})'"
