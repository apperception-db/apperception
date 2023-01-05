import collections
import collections.abc
import cv2
from datetime import datetime, timedelta

from .video import Video
from .camera_config import CameraConfig, interpolate


class VideoSkipped(Video):
    videofile: str

    def __init__(
        self, videofile: str, camera_configs: "list[CameraConfig | None]"
    ):
        self.videofile = videofile
        self._camera_configs: "list[CameraConfig | None]" = camera_configs
        config0 = camera_configs[0]
        assert config0 is not None
        self._start: "datetime" = config0.timestamp
        self._interpolated_frames: "list[CameraConfig] | None" = None
        self._num_frames: int | None = None
        self._fps: float | None = None

    @property
    def interpolated_frames(self):
        if self._interpolated_frames is None:
            num_frames, fps = self.__get_fps_and_num_frames()

            if len(self._camera_configs) == 1:
                config0 = self._camera_configs[0]
                assert config0 is not None
                self._start = config0.timestamp
                self._interpolated_frames = [config0 for _ in range(num_frames)]
            else:
                assert self._start is not None
                last_config = self._camera_configs[-1]
                assert last_config is not None
                assert last_config.timestamp > self._start + timedelta(
                    seconds=(num_frames - 1) / fps
                ), f"{last_config.timestamp} {self._start + timedelta(seconds=(num_frames - 1) / fps)}"

                self._interpolated_frames: "list[CameraConfig]" = []
                prev_idx = None
                next_idx = None
                for i, frame in enumerate(self._camera_configs):
                    if frame is None:
                        timestamp = self._start + timedelta(seconds=i / fps)
                        assert prev_idx is not None
                        if next_idx is None:
                            next_idx = _find_next_config(self._camera_configs, i)
                        
                        prev_frame = self._camera_configs[prev_idx]
                        next_frame = self._camera_configs[next_idx]

                        assert prev_frame is not None
                        assert next_frame is not None

                        self._interpolated_frames.append(interpolate(prev_frame, next_frame, timestamp))
                    else:
                        self._interpolated_frames.append(frame)
                        prev_idx = i
                        next_idx = None

        return self._interpolated_frames

    @property
    def fps(self):
        return self.__get_fps_and_num_frames()[1]

    def __getitem__(self, index):
        return self.interpolated_frames[index]

    def __iter__(self) -> "collections.abc.Iterator":
        return iter(self.interpolated_frames)

    def __len__(self):
        if self._interpolated_frames is not None:
            return len(self._interpolated_frames)

        return self.__get_fps_and_num_frames()[0]

    def __get_fps_and_num_frames(self):
        if self._num_frames is None or self._fps is None:
            cap = cv2.VideoCapture(self.videofile)
            assert cap.isOpened(), self.videofile
            self._num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self._fps = float(cap.get(cv2.CAP_PROP_FPS))
            cap.release()
            cv2.destroyAllWindows()
        return self._num_frames, self._fps


def _find_next_config(configs: "list[CameraConfig | None]", idx: int):
    for i in range(idx, len(configs)):
        if configs[i] is not None:
            return i
    raise Exception('configs should always end with a CameraConfig')
