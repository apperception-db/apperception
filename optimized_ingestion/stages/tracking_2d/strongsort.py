import torch
from pathlib import Path
from tqdm import tqdm
from typing import TYPE_CHECKING, Dict, List
from yolo_tracker.trackers.multi_tracker_zoo import create_tracker
from yolo_tracker.trackers.multi_tracker_zoo import StrongSORT as _StrongSORT
from yolo_tracker.yolov5.utils.torch_utils import select_device

from ..decode_frame.decode_frame import DecodeFrame
from ..detection_2d.detection_2d import Detection2D
from .tracking_2d import Tracking2D, Tracking2DResult

if TYPE_CHECKING:

    from ...payload import Payload


FILE = Path(__file__).resolve()
APPERCEPTION = FILE.parent.parent.parent.parent
WEIGHTS = APPERCEPTION / "weights"
reid_weights = WEIGHTS / "osnet_x0_25_msmt17.pt"


class StrongSORT(Tracking2D):
    def _run(self, payload: "Payload"):
        detections = Detection2D.get(payload)
        assert detections is not None

        images = DecodeFrame.get(payload)
        assert images is not None
        metadata: "List[Dict[int, Tracking2DResult]]" = []
        trajectories: "Dict[int, List[Tracking2DResult]]" = {}
        device = select_device("")
        strongsort = create_tracker('strongsort', reid_weights, device, False)
        assert isinstance(strongsort, _StrongSORT)
        curr_frame, prev_frame = None, None
        with torch.no_grad():
            if hasattr(strongsort, 'model'):
                if hasattr(strongsort.model, 'warmup'):
                    strongsort.model.warmup()

            assert len(detections) == len(images)
            for idx, ((det, names), im0s) in tqdm(enumerate(zip(detections, images)), total=len(images)):
                if not payload.keep[idx] or len(det) == 0:
                    metadata.append({})
                    strongsort.increment_ages()
                    prev_frame = im0s.copy()
                    continue
                im0 = im0s.copy()
                curr_frame = im0

                if hasattr(strongsort, 'tracker') and hasattr(strongsort.tracker, 'camera_update'):
                    if prev_frame is not None and curr_frame is not None:
                        strongsort.tracker.camera_update(prev_frame, curr_frame)

                confs = det[:, 4]
                output_ = strongsort.update(det.cpu(), im0)

                if len(output_) > 0:
                    labels: "Dict[int, Tracking2DResult]" = {}
                    for i, (output, conf) in enumerate(zip(output_, confs)):
                        obj_id = int(output[4])
                        cls = int(output[5])

                        bbox_left = output[0]
                        bbox_top = output[1]
                        bbox_w = output[2] - output[0]
                        bbox_h = output[3] - output[1]
                        labels[obj_id] = Tracking2DResult(
                            idx,
                            f"{idx}-{i}",
                            obj_id,
                            bbox_left,
                            bbox_top,
                            bbox_w,
                            bbox_h,
                            names[cls],
                            conf.item(),
                        )
                        if obj_id not in trajectories:
                            trajectories[obj_id] = []
                        trajectories[obj_id].append(labels[obj_id])
                    metadata.append(labels)
                else:
                    metadata.append({})
                prev_frame = curr_frame

        for trajectory in trajectories.values():
            last = len(trajectory) - 1
            for i, t in enumerate(trajectory):
                if i > 0:
                    t.prev = trajectory[i - 1]
                if i < last:
                    t.next = trajectory[i + 1]

        return None, {self.classname(): metadata}
