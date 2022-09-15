from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional
from frame import Frame, interpolate

import cv2


@dataclass
class FrameCollection:
    frames: "List[Frame]"
    video: str
    start: "Optional[datetime]"
    _interpolated_frames: "Optional[List[Frame]]"
    # metadata: "Optional[List[Any]]"

    def __init__(self, video: str, frames: "List[Frame]", start: datetime = None):
        self.video = video
        self.frames = frames
        self.start = start

    @property
    def interpolated_frames(self):
        if self._interpolated_frames is None:
            cap = cv2.VideoCapture(self.video)
            num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            cap.release()
            cv2.destroyAllWindows()

            if len(self.frames) == 1:
                self.start = self.frames[0].timestamp
                self._interpolated_frames = [self.frames[0] for _ in range(num_frames)]
            else:
                assert self.start is not None
                assert self.frames[-1].timestamp < self.start + timedelta(seconds=(num_frames - 1) / fps)

                idx = 0
                self._interpolated_frames: "List[Frame]" = []
                for i in range(num_frames):
                    t = self.start + timedelta(seconds=i / fps)
                    while self.frames[idx + 1].timestamp < t:
                        idx += 1
                    self._interpolated_frames.append(interpolate(self.frames[i], self.frames[i + 1], t))
        return self._interpolated_frames

    def __len__(self):
        cap = cv2.VideoCapture(self.video)
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        cv2.destroyAllWindows()
        return num_frames
