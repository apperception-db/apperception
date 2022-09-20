from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Optional

import cv2
from bitarray import bitarray

if TYPE_CHECKING:
    from .stages.stage import Stage
    from .video import Video


@dataclass
class Payload:
    video: "Video"
    keep: "bitarray"
    metadata: "Optional[List[Any]]"

    def __init__(
        self,
        video: "Video",
        keep: "Optional[bitarray]" = None,
        metadata: "Optional[List[Any]]" = None,
    ):
        self.keep = _default_keep(video, keep)
        self.video = video
        if metadata is not None and len(metadata) != len(video):
            raise Exception()
        self.metadata = metadata

    def filter(self, filter: "Stage"):
        keep, metadata = filter(self)

        assert keep is None or len(keep) == len(self.video)
        if keep is None:
            keep = self.keep
        else:
            keep = keep & self.keep

        assert metadata is None or len(metadata) == len(keep)
        if metadata is None:
            metadata = self.metadata
        elif self.metadata is not None:
            metadata = [_merge(m1, m2) for m1, m2 in zip(self.metadata, metadata)]

        print("Filter with: ", filter.__class__.__name__)
        print(f"  filtered frames: {sum(keep) * 100.0 / len(keep)}%")
        print(keep)

        return Payload(self.video, self.keep & keep, metadata)

    def save(self, filename: str) -> None:
        video = cv2.VideoCapture(self.video.videofile)
        images = []
        idx = 0
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break
            if not self.keep[idx]:
                frame[:, :, 2] = 255
            images.append(frame)
            idx += 1
        video.release()
        cv2.destroyAllWindows()

        height, width, _ = images[0].shape
        out = cv2.VideoWriter(
            filename, cv2.VideoWriter_fourcc(*"DIVX"), self.video.fps, (width, height)
        )
        for image in images:
            out.write(image)
        out.release()
        cv2.destroyAllWindows()


def _merge(meta1, meta2):
    if not meta1:
        return meta2
    if not meta2:
        return meta1
    return {**meta1, **meta2}


def _default_keep(video: "Video", keep: "Optional[bitarray]" = None):
    if keep is None:
        keep = bitarray(len(video))
        keep.setall(1)
    elif len(keep) != len(video):
        raise Exception()
    return keep
