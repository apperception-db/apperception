from .camera_config import CameraConfig
from .video import Video


class VideoCompressed(Video):
    videofile: "str"

    def __init__(
        self,
        videofile: str,
        camera_configs: "list[CameraConfig]"
    ):
        super().__init__(
            videofile,
            camera_configs,
            camera_configs[0].timestamp
        )

    @property
    def interpolated_frames(self):
        return self._camera_configs
