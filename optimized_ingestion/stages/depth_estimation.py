import os
import pickle
from typing import Optional, Tuple

from bitarray import bitarray
from tqdm import tqdm

from ..monodepth import monodepth
from ..payload import Payload
from .decode_frame import DecodeFrame
from .stage import Stage
from .utils.is_annotated import is_annotated


class DepthEstimation(Stage):
    def __call__(self, payload: "Payload") -> "Tuple[Optional[bitarray], Optional[list]]":
        if os.path.exists("./_DepthEstimation.pickle"):
            with open("./_DepthEstimation.pickle", "rb") as f:
                return None, pickle.load(f)

        if not is_annotated(DecodeFrame, payload):
            payload = payload.filter(DecodeFrame())

        md = monodepth()
        metadata = []
        assert payload.metadata is not None
        for k, m in tqdm([*zip(payload.keep, payload.metadata)]):
            depth = None
            if k:
                depth = md.eval(DecodeFrame.get(m))
            metadata.append({self.classname(): depth})
        # with open("./depth.pickle", "wb") as f:
        #     pickle.dump(metadata, f)
        return None, metadata
