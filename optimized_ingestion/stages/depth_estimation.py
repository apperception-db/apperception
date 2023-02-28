import numpy.typing as npt
from bitarray import bitarray
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from ..monodepth import monodepth
from .decode_frame.decode_frame import DecodeFrame
from .stage import Stage
from .utils.is_annotated import is_annotated

if TYPE_CHECKING:
    from ..payload import Payload


class DepthEstimation(Stage["npt.NDArray | None"]):
    def _run(self, payload: "Payload") -> "Tuple[Optional[bitarray], Optional[Dict[str, list]]]":
        if not is_annotated(DecodeFrame, payload):
            raise Exception()
            # payload = payload.filter(DecodeFrame())

        md = monodepth()
        assert payload.metadata is not None
        images: "List[npt.NDArray | None]" = []

        decoded_frames = DecodeFrame.get(payload)
        assert decoded_frames is not None

        for k, m in zip(payload.keep, decoded_frames):
            if k:
                images.append(m)
            else:
                images.append(None)
        metadata = md.eval_all(images)

        return None, {self.classname(): metadata}
