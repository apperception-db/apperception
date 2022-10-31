from dataclasses import dataclass
from typing import Tuple

import numpy as np
import numpy.typing as npt

from ..stage import Stage


class Tracking3D(Stage):
    pass


@dataclass
class Tracking3DResult:
    object_id: float
    point_from_camera: Tuple[float, float, float]
    point: "npt.NDArray[np.floating]"
    prev: "Tracking3DResult | None" = None
    next: "Tracking3DResult | None" = None