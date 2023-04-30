from dataclasses import dataclass, field
from typing import List

from ...types import DetectionId
from ..stage import Stage


@dataclass
class TrackingResult:
    detection_id: DetectionId
    object_id: int
    confidence: float
    next: "TrackingResult | None" = field(default=None, compare=False, repr=False)
    prev: "TrackingResult | None" = field(default=None, compare=False, repr=False)


Metadatum = List[TrackingResult]


class Tracking(Stage[Metadatum]):
    @classmethod
    def encode_json(cls, o: "Metadatum"):
        if isinstance(o, TrackingResult):
            return {
                "detection_id": tuple(o.detection_id),
                "object_id": o.object_id,
                "confidence": o.confidence,
            }
