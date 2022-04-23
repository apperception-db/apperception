from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Trajectory:
    coordinates: List[Tuple[float, float, float]]
    datetimes: List[str]
    upper_inc: bool
    lower_inc: bool
