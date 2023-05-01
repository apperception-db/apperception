from pathlib import Path

import torch
from yolo_tracker.trackers.multi_tracker_zoo import StrongSORT as _StrongSORT
from yolo_tracker.trackers.multi_tracker_zoo import create_tracker
from yolo_tracker.yolov5.utils.torch_utils import select_device

from ...cache import cache
from ...payload import Payload
from ..decode_frame.decode_frame import DecodeFrame
from ..detection_2d.detection_2d import Detection2D
from .tracking import Tracking, TrackingResult

FILE = Path(__file__).resolve()
APPERCEPTION = FILE.parent.parent.parent.parent
WEIGHTS = APPERCEPTION / "weights"
reid_weights = WEIGHTS / "osnet_x0_25_msmt17.pt"


class StrongSORT(Tracking):
    def __init__(self, cache: "bool" = True) -> None:
        super().__init__()
        self.cache = cache

    @cache
    def _run(self, payload: "Payload"):
        detections = Detection2D.get(payload)
        assert detections is not None

        images = DecodeFrame.get(payload)
        assert images is not None
        metadata: "list[list[TrackingResult]]" = []
        trajectories: "dict[int, list[TrackingResult]]" = {}
        device = select_device("")
        strongsort = create_tracker('strongsort', reid_weights, device, False)
        assert isinstance(strongsort, _StrongSORT)
        assert hasattr(strongsort, 'tracker')
        assert hasattr(strongsort.tracker, 'camera_update')
        assert hasattr(strongsort, 'model')
        assert hasattr(strongsort.model, 'warmup')
        curr_frame, prev_frame = None, None
        with torch.no_grad():
            strongsort.model.warmup()

            assert len(detections) == len(images)
            # for idx, ((det, names, dids), im0s) in tqdm(enumerate(zip(detections, images)), total=len(images)):
            for idx, ((det, names, dids), im0s) in enumerate(zip(detections, images)):
                im0 = im0s.copy()
                curr_frame = im0

                if prev_frame is not None and curr_frame is not None:
                    strongsort.tracker.camera_update(prev_frame, curr_frame, cache=self.cache)

                if not payload.keep[idx] or len(det) == 0:
                    metadata.append([])
                    strongsort.increment_ages()
                    prev_frame = curr_frame
                    continue

                confs = det[:, 4]
                output_ = strongsort.update(det.cpu(), im0)

                labels: "list[TrackingResult]" = []
                for output in output_:
                    obj_id = int(output[4])
                    det_idx = int(output[7])
                    frame_idx_offset = int(output[8])

                    if frame_idx_offset == 0:
                        did = dids[det_idx]
                        conf = confs[det_idx].item() / 10000.
                    else:
                        # We have to access detection id from previous frames.
                        # When a track is initialized, it is not in the output of strongsort.update right away.
                        # Instead if the track is confirmed, it will be in the output of strongsort.update in the next frame.
                        did = detections[idx - frame_idx_offset][2][det_idx]
                        # TODO: this is a hack.
                        # When we need to use conf, we will have to store conf from previous frames.
                        conf = -1.

                    labels.append(TrackingResult(did, obj_id, conf))
                    if obj_id not in trajectories:
                        trajectories[obj_id] = []
                    trajectories[obj_id].append(labels[-1])
                metadata.append(labels)

                prev_frame = curr_frame

        for trajectory in trajectories.values():
            for before, after in zip(trajectory[:-1], trajectory[1:]):
                before.next = after
                after.prev = before

        return None, {self.classname(): metadata}
