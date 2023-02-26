import collections
import collections.abc
import cv2
from datetime import datetime
from typing import Iterable

from ..camera_config import CameraConfig


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
        return self._camera_configs

    @property
    def fps(self):
        return self._get_fps_and_length()[1]

    def __getitem__(self, index: "int"):
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
