from dataclasses import dataclass
from typing import Tuple


@dataclass
class Point:
    point_id: str
    object_id: str
    coordinate: Tuple[float, float, float]
    time: float
    point_type: str
