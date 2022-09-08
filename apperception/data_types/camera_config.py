from dataclasses import dataclass
from typing import List, Tuple
import numpy.typing as npt

Float3 = Tuple[float, float, float]
Float4 = Tuple[float, float, float, float]


@dataclass(frozen=True)
class CameraConfig:
    frame_id: str
    frame_num: int
    filename: str
    camera_translation: npt.NDArray  # float[3]
    camera_rotation: npt.NDArray  # float[4]
    camera_intrinsic: List[List[float]]  # float[3][3]
    ego_translation: List[float]  # float[3]
    ego_rotation: List[float]  # float[4]
    timestamp: str
    cameraHeading: float
    egoHeading: float
