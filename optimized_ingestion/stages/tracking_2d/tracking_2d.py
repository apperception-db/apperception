from dataclasses import dataclass, field
from typing import Dict

import numpy as np

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
    confidence: "float | np.float32"
    next: "Tracking2DResult | None" = field(default=None, compare=False, repr=False)
    prev: "Tracking2DResult | None" = field(default=None, compare=False, repr=False)


Metadatum = Dict[int, Tracking2DResult]


class Tracking2D(Stage[Metadatum]):
    @classmethod
    def encode_json(cls, o: "Metadatum"):
        if isinstance(o, Tracking2DResult):
            assert isinstance(o.frame_idx, int), type(o.frame_idx)
            assert isinstance(o.object_id, int), type(o.object_id)
            assert isinstance(o.bbox_left, float), type(o.bbox_left)
            assert isinstance(o.bbox_top, float), type(o.bbox_top)
            assert isinstance(o.bbox_w, float), type(o.bbox_w)
            assert isinstance(o.bbox_h, float), type(o.bbox_h)
            assert isinstance(o.object_type, str), type(o.object_type)
            assert isinstance(o.confidence, (float, np.floating)), type(o.confidence)

            return {
                "frame_idx": o.frame_idx,
                "detection_id": tuple(o.detection_id),
                "object_id": o.object_id,
                "bbox_left": o.bbox_left,
                "bbox_top": o.bbox_top,
                "bbox_w": o.bbox_w,
                "bbox_h": o.bbox_h,
                "object_type": o.object_type,
                "confidence": float(o.confidence)
            }
