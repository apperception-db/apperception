from typing import Optional, Tuple

from bitarray import bitarray

from ..monodepth import monodepth
from ..payload import Payload
from .decode_frame_filter import DecodeFrameFilter
from .filter import Filter
from .utils.is_filtered import is_filtered


class DepthEstimationFilter(Filter):
    def __call__(self, payload: "Payload") -> "Tuple[Optional[bitarray], Optional[list]]":
        if not is_filtered(DecodeFrameFilter, payload):
            payload = payload.filter(DecodeFrameFilter())

        md = monodepth()
        metadata = []
        for k, m in zip(payload.keep, payload.metadata):
            depth = None
            if k:
                depth = md.eval(DecodeFrameFilter.get(m))
            metadata.append({self.classname(): depth})
        return None, metadata
