from dataclasses import dataclass, field
from typing import Dict

from ...types import DetectionId
from ..stage import Stage


@dataclass
class Tracking2DResult:
    frame_idx: int
    detection_id: DetectionId
    object_id: int
    bbox_left: float
    bbox_top: float
    bbox_w: float
    bbox_h: float
    object_type: str
    confidence: float
    next: "Tracking2DResult | None" = field(default=None, compare=False, repr=False)
    prev: "Tracking2DResult | None" = field(default=None, compare=False, repr=False)


Metadatum = Dict[int, Tracking2DResult]


class Tracking2D(Stage[Metadatum]):
    pass
