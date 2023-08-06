from ..camera_config import CameraConfig, has_config, interpolate
from .video import Video


class VideoSkipped(Video):
    videofile: "str"

    def __init__(self, videofile: str, camera_configs: "list[CameraConfig]"):
        super().__init__(videofile, camera_configs, camera_configs[0].timestamp)

    @property
    def interpolated_frames(self):
        if not hasattr(self, "_interpolated_frames"):
            length, fps, _ = self._get_props()

            if len(self._camera_configs) == 1:
                config0 = self._camera_configs[0]
                self._start = config0.timestamp
                self._interpolated_frames = [config0 for _ in range(length)]
            else:
                assert self._start is not None
                last_config = self._camera_configs[-1]
                assert has_config(last_config)
                # assert round((last_config.timestamp - self._start).total_seconds() * fps) == length, ((last_config.timestamp - self._start).total_seconds(), fps, length)

                self._interpolated_frames: "list[CameraConfig]" = []
                prev_frame = None
                next_frame = None
                for i, frame in enumerate(self._camera_configs):
                    if has_config(frame):
                        self._interpolated_frames.append(frame)
                        prev_frame = frame
                        next_frame = None
                    else:
                        timestamp = frame.timestamp
                        assert prev_frame is not None, i
                        if next_frame is None:
                            next_frame = _find_first_config(self._camera_configs[i:])

                        self._interpolated_frames.append(
                            interpolate(prev_frame, next_frame, timestamp)
                        )
                assert len(self._interpolated_frames) == len(self._camera_configs)

        return self._interpolated_frames


def _find_first_config(configs: "list[CameraConfig]"):
    for c in configs:
        if has_config(c):
            return c
    raise Exception("configs should always end with a CameraConfig")
