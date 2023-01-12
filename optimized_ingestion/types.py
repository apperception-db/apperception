from typing import NamedTuple, Tuple


class DetectionId(NamedTuple):
    frame_idx: int
    obj_order: int


Float2 = Tuple[float, float]
Float22 = Tuple[Float2, Float2]

Float3 = Tuple[float, float, float]
Float33 = Tuple[Float3, Float3, Float3]