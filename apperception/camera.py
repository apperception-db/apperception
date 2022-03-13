import uuid
from dataclasses import dataclass
from typing import List, Optional

from camera_config import CameraConfig


@dataclass
class Camera:
    id: str
    configs: List[CameraConfig]

    def __init__(self, config: List[CameraConfig], id: Optional[str] = None):
        self.id = str(uuid.uuid4()) if id is None else id
        self.configs = config
