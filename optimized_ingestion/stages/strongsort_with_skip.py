import copy
from pathlib import Path

import numpy.typing as npt
import torch

from ..modules.yolo_tracker.trackers.multi_tracker_zoo import StrongSORT, create_tracker
from ..modules.yolo_tracker.yolov5.utils.torch_utils import select_device
from ..payload import Payload
from .decode_frame.decode_frame import DecodeFrame
from .detection_2d.detection_2d import Detection2D, Metadatum
from .stage import Stage

FILE = Path(__file__).resolve()
APPERCEPTION = FILE.parent.parent.parent
WEIGHTS = APPERCEPTION / "weights"
reid_weights = WEIGHTS / "osnet_x0_25_msmt17.pt"


class StrongSORTWithSkip(Stage["list[dict[str, list[int]]]"]):
    def __init__(self, skip: int = 35):
        self.skip = skip

    def _run(self, payload: "Payload"):
        detections = Detection2D.get(payload)
        assert detections is not None

        images = DecodeFrame.get(payload)
        assert images is not None

        device = select_device("")
        strongsort = create_tracker('strongsort', reid_weights, device, False)
        assert isinstance(strongsort, StrongSORT)
        assert hasattr(strongsort, 'tracker')
        assert hasattr(strongsort.tracker, 'camera_update')

        trackings: "list[list[dict[str, list[int]]]]" = []
        prev_frame = None
        with torch.no_grad():
            if hasattr(strongsort, 'model') and hasattr(strongsort.model, 'warmup'):
                strongsort.model.warmup()

            assert len(detections) == len(images)
            for idx, ((det, names, dids), im0s) in enumerate(zip(detections, images)):
                # print(idx)
                _trackings: "list[dict[str, list[int]]]" = []
                prev_frame = process_one_frame(strongsort, detections, _trackings, idx, Metadatum(det, names, dids), im0s, prev_frame)
                if idx % 10 == 0:
                    _strongsort = copy.deepcopy(strongsort)
                    _prev_frame = None
                    if prev_frame is not None:
                        _prev_frame = prev_frame.copy()
                    next_idx = idx + 1
                    for _idx, (_det, _names, _dids), _im0s in zip(
                        range(next_idx, next_idx + self.skip),
                        detections[next_idx:][:self.skip],
                        images[next_idx:][:self.skip]
                    ):
                        # print("  ", _idx)
                        __strongsort = copy.deepcopy(_strongsort)
                        __trackings: "list[dict[str, list[int]]]" = []
                        for __idx in range(5):
                            process_one_frame(__strongsort, detections, __trackings, _idx + __idx, Metadatum())
                        process_one_frame(__strongsort, detections, _trackings, _idx, Metadatum(_det, _names, _dids), _im0s, _prev_frame)
                        # TODO: process nother 5 frames

                        _curr_frame = _im0s.copy()
                        if _prev_frame is not None and _curr_frame is not None:
                            _strongsort.tracker.camera_update(_prev_frame, _curr_frame, cache=True)
                        _strongsort.increment_ages()

                        _prev_frame = _curr_frame

                trackings.append(_trackings)

        return None, {self.classname(): trackings}


def process_one_frame(
    ss: "StrongSORT",
    detections: "list[Metadatum]",
    _trackings: "list[dict[str, list[int]]]",
    idx: int,
    detection: "Metadatum",
    im0s: "npt.NDArray",
    prev_frame: "npt.NDArray | None",
) -> "npt.NDArray":
    det, _, dids = detection
    im0 = im0s.copy()
    curr_frame = im0

    if prev_frame is not None and curr_frame is not None:
        ss.tracker.camera_update(prev_frame, curr_frame, cache=True)

    output_ = ss.update(det.cpu(), im0)

    t2ds: "dict[str, list[int]]" = {}
    for output in output_:
        det_idx = int(output[7])
        frame_idx_offset = int(output[8])
        if frame_idx_offset == 0:
            did = dids[det_idx]
        else:
            did = detections[idx - frame_idx_offset][2][det_idx]
        assert repr(did) not in t2ds, (did, t2ds, output_)
        t2ds[repr(did)] = output.tolist()

    _trackings.append(t2ds)
    prev_frame = curr_frame

    return prev_frame
