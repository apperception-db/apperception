from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class TrackedObject():
    object_type: str
    bboxes: List[Tuple[Tuple[int, int], Tuple[int, int]]] = field(default_factory=list)
    tracked_cnt: List[int] = field(default_factory=list)
