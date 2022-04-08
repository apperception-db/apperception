from dataclasses import dataclass
from typing import List, Tuple

Float3 = Tuple[float, float, float]
Float4 = Tuple[float, float, float, float]


@dataclass(frozen=True)
class CameraConfig:
    frame_id: str
    frame_num: int
    filename: str
    camera_translation: List[float]  # float[3]
    camera_rotation: List[float]  # float[4]
    camera_intrinsic: List[List[float]]  # float[3][3]
    ego_translation: List[float]  # float[3]
    ego_rotation: List[float]  # float[4]
    timestamp: str
    heading: float


def fetch_camera_config(scene_name: str, sample_data):
    all_frames = sample_data[
        (sample_data["scene_name"] == scene_name)
        & (
            sample_data["filename"].str.contains("/CAM_FRONT/", regex=False)
        )  # temporary while debugging
    ].sort_values(by="frame_order")

    return [
        CameraConfig(
            frame_id=frame.token,
            frame_num=frame.frame_order,
            filename=frame.filename,
            camera_translation=frame.camera_translation,
            camera_rotation=frame.camera_rotation,
            camera_intrinsic=frame.camera_intrinsic,
            ego_translation=frame.ego_translation,
            ego_rotation=frame.ego_rotation,
            timestamp=frame.timestamp,
            heading=frame.heading,
        )
        for frame in all_frames.itertuples(index=False)
    ]
