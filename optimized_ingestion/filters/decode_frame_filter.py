from typing import  List, Optional, Tuple

import cv2
from bitarray import bitarray

from ..payload import Payload
from .filter import Filter


class DecodeFrameFilter(Filter):
    def __call__(self, payload: "Payload") -> "Tuple[Optional[bitarray], Optional[list]]":
        metadata: "List[dict]" = []

        # TODO: only decode filtered frames
        video = cv2.VideoCapture(payload.frames.video)
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break
            metadata.append({DecodeFrameFilter.__class__.__name__: frame})
        assert len(metadata) == len(payload.frames)
        video.release()
        cv2.destroyAllWindows()

        return None, metadata
