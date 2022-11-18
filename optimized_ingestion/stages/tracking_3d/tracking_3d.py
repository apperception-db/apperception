import numpy as np
import numpy.typing as npt
from typing import NamedTuple, Tuple

from ..stage import Stage


class Tracking3D(Stage):
    pass


class Tracking3DResult(NamedTuple):
    object_id: float
    point_from_camera: Tuple[float, float, float]
    point: "npt.NDArray[np.floating]"
    prev: "Tracking3DResult | None" = None
    next: "Tracking3DResult | None" = None
