import numpy.typing as npt
import os
import pickle
from bitarray import bitarray
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from ..monodepth import monodepth
from .decode_frame import DecodeFrame
from .stage import Stage
from .utils.is_annotated import is_annotated

if TYPE_CHECKING:
    from ..payload import Payload


class DepthEstimation(Stage):
    def _run(self, payload: "Payload") -> "Tuple[Optional[bitarray], Optional[Dict[str, list]]]":
        if os.path.exists("./_DepthEstimation.pickle"):
            with open("./_DepthEstimation.pickle", "rb") as f:
                return None, {self.classname(): pickle.load(f)}

        if not is_annotated(DecodeFrame, payload):
            raise Exception()
            # payload = payload.filter(DecodeFrame())

        md = monodepth()
        assert payload.metadata is not None
        images: "List[npt.NDArray | None]" = []
        for k, m in zip(payload.keep, DecodeFrame.get(payload.metadata)):
            if k:
                images.append(m)
            else:
                images.append(None)
        metadata = md.eval_all(images)

        with open("./_DepthEstimation.pickle", "wb") as f:
            pickle.dump(metadata, f)
        return None, {self.classname(): metadata}
