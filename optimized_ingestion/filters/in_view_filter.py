from typing import Optional, Tuple

from bitarray import bitarray

from apperception.database import database

from .filter import Filter
from ..payload import Payload


class InViewFilter(Filter):
    def __init__(self, distance: float, segment_type: str) -> None:
        self.distance = distance
        self.segment_type = segment_type

    def __call__(self, payload: "Payload") -> "Tuple[Optional[bitarray], Optional[list]]":
        keep = bitarray()
        for _frame in payload.frames:
            point = f"'POINT ({' '.join([*map(str, _frame.ego_translation)])})'"
            query = (
                f"SELECT TRUE WHERE minDistance({point}, '{self.segment_type}') < {self.distance}"
            )
            result = database._execute_query(query)
            keep.append(bool(result))

        return keep, None
