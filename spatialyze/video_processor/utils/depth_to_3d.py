from math import sqrt

import numpy as np
import numpy.typing as npt

from ..types import Float3, Float33


def depth_to_3d(
    x: float,
    y: float,
    depth: float,
    intrinsic: "npt.NDArray[np.float32] | Float33",
) -> "Float3":
    [[fx, _, x0], [_, fy, y0], [_, _, s]] = intrinsic

    unit_x: float = (s * x - x0) / fx
    unit_y: float = (s * y - y0) / fy

    Z = depth / sqrt(1 + (unit_x**2) + (unit_y**2))
    X = unit_x * Z
    Y = unit_y * Z
    return X, Y, Z
