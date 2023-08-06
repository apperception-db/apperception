import datetime
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import numpy.typing as npt

from ...types import DetectionId
from ..stage import Stage


@dataclass
class Tracking3DResult:
    frame_idx: int
    detection_id: DetectionId
    object_id: float
    point_from_camera: Tuple[float, float, float]
    point: "npt.NDArray[np.floating]"
    bbox_left: float
    bbox_top: float
    bbox_w: float
    bbox_h: float
    object_type: str
    timestamp: datetime.datetime
    prev: "Tracking3DResult | None" = None
    next: "Tracking3DResult | None" = None


Metadatum = Dict[int, Tracking3DResult]


class Tracking3D(Stage[Metadatum]):
    @classmethod
    def encode_json(cls, o: "Any"):
        if isinstance(o, Tracking3DResult):
            return {
                "frame_idx": o.frame_idx,
                "detection_id": tuple(o.detection_id),
                "object_id": o.object_id,
                "point_from_camera": o.point_from_camera,
                "point": o.point.tolist(),
                "bbox_left": o.bbox_left,
                "bbox_top": o.bbox_top,
                "bbox_w": o.bbox_w,
                "bbox_h": o.bbox_h,
                "object_type": o.object_type,
                "timestamp": str(o.timestamp),
            }
