import collections.abc

import cv2
import numpy.typing as npt


def iterate_video(cap: "cv2.VideoCapture"):
    return VideoIterator(cap)


class VideoIterator(collections.abc.Iterator, collections.abc.Sized):
    def __init__(self, cap: "cv2.VideoCapture"):
        self._n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._count = 0
        self._cap = cap

    def __len__(self) -> int:
        return self._n

    def __iter__(self):
        return self

    def __next__(self) -> "npt.NDArray":
        if not self._cap.isOpened():
            raise Exception()

        ret, frame = self._cap.read()
        self._count += 1
        if ret:
            return frame

        assert self._count == self._n + 1, f"count: {self._count}, n: {self._n}"
        self._cap.release()
        cv2.destroyAllWindows()
        raise StopIteration()
