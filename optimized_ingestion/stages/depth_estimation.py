from typing import Optional, Tuple

from bitarray import bitarray

from ..monodepth import monodepth
from ..payload import Payload
from .decode_frame import DecodeFrame
from .stage import Stage
from .utils.is_annotated import is_annotated


class DepthEstimation(Stage):
    def __call__(self, payload: "Payload") -> "Tuple[Optional[bitarray], Optional[list]]":
        if not is_annotated(DecodeFrame, payload):
            payload = payload.filter(DecodeFrame())

        md = monodepth()
        metadata = []
        for k, m in zip(payload.keep, payload.metadata):
            depth = None
            if k:
                depth = md.eval(DecodeFrame.get(m))
            metadata.append({self.classname(): depth})
        return None, metadata
