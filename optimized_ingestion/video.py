import collections
import collections.abc
import cv2
from datetime import datetime, timedelta
from typing import Iterable

from .camera_config import CameraConfig, interpolate


class Video(Iterable["CameraConfig"]):
    videofile: str

    def __init__(
        self,
        videofile: str,
        camera_configs: "list[CameraConfig]",
        start: "datetime | None" = None,
    ):
        self.videofile = videofile
        self._camera_configs: "list[CameraConfig]" = camera_configs
        self._start: "datetime | None" = start
        self._length: "int | None" = None
        self._fps: "float | None" = None

    @property
    def interpolated_frames(self):
        if not hasattr(self, "_interpolated_frames"):
            num_frames, fps = self._get_fps_and_length()

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

    @property
    def fps(self):
        return self._get_fps_and_length()[1]

    def __getitem__(self, index):
        return self.interpolated_frames[index]

    def __iter__(self) -> "collections.abc.Iterator":
        return iter(self.interpolated_frames)

    def __len__(self):
        return self._get_fps_and_length()[0]

    def _get_fps_and_length(self):
        if self._length is None or self._fps is None:
            cap = cv2.VideoCapture(self.videofile)
            assert cap.isOpened(), self.videofile
            self._length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self._fps = float(cap.get(cv2.CAP_PROP_FPS))
            cap.release()
            cv2.destroyAllWindows()
        return self._length, self._fps
