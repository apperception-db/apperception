import datetime
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass
from typing import Dict, Tuple

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
    pass
