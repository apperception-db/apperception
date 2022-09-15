from dataclasses import dataclass
from typing import Any, List, Optional, TYPE_CHECKING

from bitarray import bitarray

if TYPE_CHECKING:
    from frame_collection import FrameCollection


@dataclass
class Payload:
    frames: "FrameCollection"
    keep: "bitarray"
    metadata: "Optional[List[Any]]"

    def __init__(self, frames: "FrameCollection", keep: "Optional[bitarray]" = None, metadata: "Optional[List[Any]]" = None):
        self.keep = _default_keep(frames, keep)
        self.frames = frames
        if metadata is not None and len(metadata) != len(frames):
            raise Exception()
        self.metadata = metadata

    def filter(self, filter):
        keep, metadata = filter(self)

        assert len(metadata) == len(self.frames)

        if self.metadata is not None:
            metadata = [
                _merge(m1, m2)
                for m1, m2
                in zip(self.metadata, metadata)
            ]

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
