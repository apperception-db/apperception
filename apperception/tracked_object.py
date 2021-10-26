from dataclasses import dataclass, field
from typing import List
from bounding_box import BoundingBox


@dataclass
class TrackedObject():
    object_type: str
    bboxes: List[BoundingBox] = field(default_factory=list)
    tracked_cnt: List[int] = field(default_factory=list)
