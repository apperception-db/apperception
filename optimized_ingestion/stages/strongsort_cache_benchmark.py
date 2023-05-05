from pathlib import Path

import torch

from ..modules.yolo_tracker.trackers.multi_tracker_zoo import StrongSORT, create_tracker
from ..modules.yolo_tracker.yolov5.utils.torch_utils import select_device
from ..payload import Payload
from .decode_frame.decode_frame import DecodeFrame
from .detection_2d.detection_2d import Detection2D
from .stage import Stage

FILE = Path(__file__).resolve()
APPERCEPTION = FILE.parent.parent.parent
WEIGHTS = APPERCEPTION / "weights"
reid_weights = WEIGHTS / "osnet_x0_25_msmt17.pt"


class StrongSORTCacheBenchmark(Stage["dict[str, list[int]]"]):
    def __init__(self, cache: bool):
        self.cache = cache
        self.benchmark_cache = []

    def _run(self, payload: "Payload"):
        detections = Detection2D.get(payload)
        assert detections is not None

        images = DecodeFrame.get(payload)
        assert images is not None

        device = select_device("")
        if StrongSORTCacheBenchmark.progress:
            print(device)
        strongsort = create_tracker('strongsort', reid_weights, device, False)
        assert isinstance(strongsort, StrongSORT)
        assert hasattr(strongsort, 'tracker')
        assert hasattr(strongsort.tracker, 'camera_update')

        trackings: "list[dict[str, list[int]]]" = []
        curr_frame, prev_frame = None, None
        with torch.no_grad():
            if hasattr(strongsort, 'model'):
                if hasattr(strongsort.model, 'warmup'):
                    strongsort.model.warmup()

            assert len(detections) == len(images)
            for (det, _, dids), im0s in StrongSORTCacheBenchmark.tqdm(zip(detections, images), total=len(detections)):
                im0 = im0s.copy()
                curr_frame = im0

                if prev_frame is not None and curr_frame is not None:
                    strongsort.tracker.camera_update(prev_frame, curr_frame, cache=self.cache)

                output_, dids_ = strongsort.update(det.cpu(), dids, im0)

                t2ds: "dict[str, list[int]]" = {}
                for output, did in zip(output_, dids_):
                    assert repr(did) not in t2ds, (did, t2ds, output_)
                    t2ds[repr(did)] = output.tolist()

                trackings.append(t2ds)
                prev_frame = curr_frame

            _tracksings: "list[dict[str, list[int]]]" = [{} for _ in range(len(trackings))]

        return None, {self.classname(): trackings}
