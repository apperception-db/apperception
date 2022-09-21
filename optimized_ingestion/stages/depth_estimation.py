import os
import pickle
from typing import List, TYPE_CHECKING, Optional, Tuple

from bitarray import bitarray
import numpy.typing as npt

from ..monodepth import monodepth
from .decode_frame import DecodeFrame
from .stage import Stage

# from .utils.is_annotated import is_annotated
if TYPE_CHECKING:
    from ..payload import Payload


class DepthEstimation(Stage):
    def __call__(self, payload: "Payload") -> "Tuple[Optional[bitarray], Optional[list]]":
        if os.path.exists("./_DepthEstimation.pickle"):
            with open("./_DepthEstimation.pickle", "rb") as f:
                return None, pickle.load(f)

        # if not is_annotated(DecodeFrame, payload):
        #     payload = payload.filter(DecodeFrame())

        md = monodepth()
        assert payload.metadata is not None
        images: "List[npt.NDArray | None]" = []
        for k, m in zip(payload.keep, payload.metadata):
            if k:
                images.append(DecodeFrame.get(m))
            else:
                images.append(None)
        metadata = [{self.classname(): depth} for depth in md.eval_all(images)]

        with open("./_DepthEstimation.pickle", "wb") as f:
            pickle.dump(metadata, f)
        return None, metadata
