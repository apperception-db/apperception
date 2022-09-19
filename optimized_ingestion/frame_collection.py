import collections
import collections.abc
from datetime import datetime, timedelta
from typing import List, Optional

import cv2

from .frame import Frame, interpolate


class FrameCollection(collections.abc.Iterable):
    frames: "List[Frame]"
    video: str
    start: "Optional[datetime]"

    def __init__(self, video: str, frames: "List[Frame]", start: datetime = None):
        self.video = video
        self.frames = frames
        self.start = start
        self._interpolated_frames: "Optional[List[Frame]]" = None
        self.__num_frames: int = None
        self.__fps: float = None

    @property
    def interpolated_frames(self):
        if self._interpolated_frames is None:
            num_frames, fps = self.__get_fps_and_num_frames()

            if len(self.frames) == 1:
                self.start = self.frames[0].timestamp
                self._interpolated_frames = [self.frames[0] for _ in range(num_frames)]
            else:
                assert self.start is not None
                assert self.frames[-1].timestamp > self.start + timedelta(
                    seconds=(num_frames - 1) / fps
                ), f"{self.frames[-1].timestamp} {self.start + timedelta(seconds=(num_frames - 1) / fps)}"

                idx = 0
                self._interpolated_frames: "List[Frame]" = []
                for i in range(num_frames):
                    t = self.start + timedelta(seconds=i / fps)
                    while self.frames[idx + 1].timestamp < t:
                        idx += 1
                    self._interpolated_frames.append(
                        interpolate(self.frames[idx], self.frames[idx + 1], t)
                    )
        return self._interpolated_frames

    def __getitem__(self, index):
        return self.interpolated_frames[index]

    def __iter__(self) -> "collections.abc.Iterator[Frame]":
        return iter(self.interpolated_frames)

    def __len__(self):
        if self._interpolated_frames is not None:
            return len(self._interpolated_frames)

        return self.__get_fps_and_num_frames()[0]

    def __get_fps_and_num_frames(self):
        if self.__num_frames is None or self.__fps is None:
            cap = cv2.VideoCapture(self.video)
            self.__num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.__fps = float(cap.get(cv2.CAP_PROP_FPS))
            cap.release()
            cv2.destroyAllWindows()
        return self.__num_frames, self.__fps