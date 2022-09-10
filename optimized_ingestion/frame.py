from typing import Any, List, Tuple

Float3 = Tuple[float, float, float]
Float4 = Tuple[float, float, float, float]


class Frame:
    camera_id: str
    frame_id: str
    frame_num: int
    filename: str
    camera_translation: List[float]  # float[3]
    camera_rotation: List[float]  # float[4]
    camera_intrinsic: List[List[float]]  # float[3][3]
    ego_translation: List[float]  # float[3]
    ego_rotation: List[float]  # float[4]
    timestamp: str
    camera_heading: float
    ego_heading: float
    camera_translation_abs: List[float]  # float[3]
    road_direction: float

    def __init__(self, camera_config: Tuple[Any, ...]) -> None:
        (
            self.camera_id,
            self.frame_id,
            self.frame_num,
            self.filename,
            self.camera_translation,
            self.camera_rotation,
            self.camera_intrinsic,
            self.ego_translation,
            self.ego_rotation,
            self.timestamp,
            self.camera_heading,
            self.ego_heading,
            self.road_direction,
        ) = camera_config

    def get_tuple(self) -> Tuple[Any, ...]:
        return (
            self.camera_id,
            self.frame_id,
            self.frame_num,
            self.filename,
            self.camera_translation,
            self.camera_rotation,
            self.camera_intrinsic,
            self.ego_translation,
            self.ego_rotation,
            self.timestamp,
            self.camera_heading,
            self.ego_heading,
            self.road_direction
        )
