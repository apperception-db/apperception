from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
from yolo_tracker.trackers.multi_tracker_zoo import StrongSORT, create_tracker
from yolo_tracker.yolov5.utils.torch_utils import select_device

from ..payload import Payload
from .decode_frame.decode_frame import DecodeFrame
from .detection_2d.detection_2d import Detection2D
from .stage import Stage

FILE = Path(__file__).resolve()
APPERCEPTION = FILE.parent.parent.parent
WEIGHTS = APPERCEPTION / "weights"
reid_weights = WEIGHTS / "osnet_x0_25_msmt17.pt"


class StrongSORTCacheBenchmark(Stage["tuple[dict[str, list[float]], dict[str, list[float]]]"]):
    def __init__(self):
        self.benchmark_cache = []

    def _run(self, payload: "Payload"):
        all_trackings: "list[list[dict[str, npt.NDArray[np.int32]]]]" = []
        for cache in [False, True]:
            detections = Detection2D.get(payload)
            assert detections is not None

            images = DecodeFrame.get(payload)
            assert images is not None

            device = select_device("")
            strongsort = create_tracker('strongsort', reid_weights, device, False)
            assert isinstance(strongsort, StrongSORT)
            assert hasattr(strongsort, 'tracker')
            assert hasattr(strongsort.tracker, 'camera_update')

            trackings: "list[dict[str, npt.NDArray[np.int32]]]" = []
            curr_frame, prev_frame = None, None
            with torch.no_grad():
                if hasattr(strongsort, 'model'):
                    if hasattr(strongsort.model, 'warmup'):
                        strongsort.model.warmup()

                assert len(detections) == len(images)
                for idx, ((det, names, dids), im0s) in enumerate(zip(detections, images)):
                    if not payload.keep[idx] or len(det) == 0:
                        trackings.append({})
                        strongsort.increment_ages()
                        prev_frame = im0s.copy()
                        continue

                    im0 = im0s.copy()
                    curr_frame = im0

                    if prev_frame is not None and curr_frame is not None:
                        strongsort.tracker.camera_update(prev_frame, curr_frame, cache=cache)

                    output_ = strongsort.update(det.cpu(), im0)

                    t2ds: "dict[str, npt.NDArray[np.int32]]" = {}
                    for output in output_:
                        det_idx = int(output[7])
                        frame_idx_offset = int(output[8])
                        if frame_idx_offset == 0:
                            did = dids[det_idx]
                        else:
                            did = detections[idx - frame_idx_offset][2][det_idx]
                        assert repr(did) not in t2ds, (did, t2ds, output_)
                        t2ds[repr(did)] = output

                    trackings.append(t2ds)
                    prev_frame = curr_frame
                all_trackings.append(trackings)

        metadata = [*zip(*all_trackings)]
        return None, {self.classname(): metadata}
