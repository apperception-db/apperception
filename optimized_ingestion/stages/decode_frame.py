from typing import List, Optional, Tuple

import cv2
from bitarray import bitarray

from ..payload import Payload
from .stage import Stage


class DecodeFrame(Stage):
    def __call__(self, payload: "Payload") -> "Tuple[Optional[bitarray], Optional[list]]":
        metadata: "List[dict]" = []
        print('decoding')

        # TODO: only decode filtered frames
        video = cv2.VideoCapture(payload.video.videofile)
        while video.isOpened():
            print(len(metadata))
            ret, frame = video.read()
            print(ret)
            if not ret:
                break
            metadata.append({self.classname(): frame})
        assert len(metadata) == len(payload.video)
        video.release()
        cv2.destroyAllWindows()
        print('decoded')

        return None, metadata
