from typing import TYPE_CHECKING, List, Optional, Tuple

import cv2
from bitarray import bitarray
from tqdm import tqdm

from .stage import Stage
if TYPE_CHECKING:
    from ..payload import Payload


class DecodeFrame(Stage):
    def __call__(self, payload: "Payload") -> "Tuple[Optional[bitarray], Optional[list]]":
        metadata: "List[dict]" = []

        # TODO: only decode filtered frames
        video = cv2.VideoCapture(payload.video.videofile)
        n_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        for _ in tqdm(range(n_frames)):
            if not video.isOpened():
                break
            frame = video.read()[1]
            metadata.append({self.classname(): frame})
        assert len(metadata) == len(payload.video)
        assert not video.read()[0]
        video.release()
        cv2.destroyAllWindows()

        return None, metadata
