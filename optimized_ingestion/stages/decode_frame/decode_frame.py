import cv2
import numpy as np
from tqdm import tqdm
from typing import TYPE_CHECKING, List

from ...cache import cache
from ..stage import Stage

if TYPE_CHECKING:
    import numpy.typing as npt

    from ...payload import Payload


class DecodeFrame(Stage[np.ndarray]):
    @cache
    def _run(self, payload: "Payload"):
        metadata: "List[npt.NDArray]" = []

        video = cv2.VideoCapture(payload.video.videofile)
        n_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        for _ in tqdm(range(n_frames)):
            if not video.isOpened():
                break
            frame = video.read()[1]
            metadata.append(frame)
        assert len(metadata) == len(payload.video)
        assert not video.read()[0]
        video.release()
        cv2.destroyAllWindows()

        return None, {self.classname(): metadata}