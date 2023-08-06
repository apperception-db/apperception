from datetime import datetime, timedelta

from ..camera_config import CameraConfig, interpolate
from .video import Video


class InterpolableVideo(Video):
    def __init__(
        self, videofile: str, camera_configs: "list[CameraConfig]", start: "datetime | None" = None
    ):
        super().__init__(videofile, camera_configs, start)

    @property
    def interpolated_frames(self):
        if not hasattr(self, "_interpolated_frames"):
            num_frames, fps, _ = self._get_props()

            if len(self._camera_configs) == 1:
                self._start = self._camera_configs[0].timestamp
                self._interpolated_frames = [self._camera_configs[0] for _ in range(num_frames)]
            else:
                assert self._start is not None
                assert self._camera_configs[-1].timestamp > self._start + timedelta(
                    seconds=(num_frames - 1) / fps
                ), f"{self._camera_configs[-1].timestamp} {self._start + timedelta(seconds=(num_frames - 1) / fps)}"

                idx = 0
                self._interpolated_frames: "list[CameraConfig]" = []
                for i in range(num_frames):
                    t = self._start + timedelta(seconds=i / fps)
                    while self._camera_configs[idx + 1].timestamp < t:
                        idx += 1
                    self._interpolated_frames.append(
                        interpolate(self._camera_configs[idx], self._camera_configs[idx + 1], t)
                    )
        return self._interpolated_frames
