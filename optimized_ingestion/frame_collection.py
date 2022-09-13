from dataclasses import dataclass
from typing import List
from frame import Frame

import cv2


@dataclass
class FrameCollection:
    frames: "List[Frame]"
    video: str

    def __init__(self, video: str, frames: "List[Frame]"):
        self.video = video
        cap = cv2.VideoCapture(video)
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))

        if len(frames) == 1:
            # TODO: copy frames[0] to all the frames in video
            pass
        else:
            # TODO: interpolate each frame to fit every frames in video
            pass
