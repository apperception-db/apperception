from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Optional

from bitarray import bitarray

if TYPE_CHECKING:
    from frame_collection import FrameCollection


@dataclass
class Payload:
    frames: "FrameCollection"
    keep: "bitarray"
    metadata: "Optional[List[Any]]"

    def __init__(
        self,
        frames: "FrameCollection",
        keep: "Optional[bitarray]" = None,
        metadata: "Optional[List[Any]]" = None,
    ):
        self.keep = _default_keep(frames, keep)
        self.frames = frames
        if metadata is not None and len(metadata) != len(frames):
            raise Exception()
        self.metadata = metadata

    def filter(self, filter):
        keep, metadata = filter(self)

        assert len(keep) == len(self.frames)
        assert metadata is None or len(metadata) == len(keep)

        if metadata is None:
            metadata = self.metadata
        elif self.metadata is not None:
            metadata = [_merge(m1, m2) for m1, m2 in zip(self.metadata, metadata)]

        keep = keep & self.keep
        print("Filter with: ", filter.__class__.__name__)
        print(f"  filtered frames: {sum(keep) * 100.0 / len(keep)}%")
        print(keep)

        return Payload(self.frames, self.keep & keep, metadata)


def _merge(meta1, meta2):
    if not meta1:
        return meta2
    if not meta2:
        return meta1
    return {**meta1, **meta2}


def _default_keep(frames: "FrameCollection", keep: "Optional[bitarray]" = None):
    if keep is None:
        keep = bitarray(len(frames))
        keep.setall(1)
    elif len(keep) != len(frames):
        raise Exception()
    return keep
