import numpy as np
import numpy.typing as npt
from dataclasses import dataclass
from typing import Dict, Tuple

from ..stage import Stage


@dataclass
class Tracking3DResult:
    detection_id: str
    object_id: float
    point_from_camera: Tuple[float, float, float]
    point: "npt.NDArray[np.floating]"
    prev: "Tracking3DResult | None" = None
    next: "Tracking3DResult | None" = None

Metadatum = Dict[float, Tracking3DResult]


class Tracking3D(Stage[Metadatum]):
    pass
