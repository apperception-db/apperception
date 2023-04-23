import torch
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List
from yolo_tracker.trackers.multi_tracker_zoo import StrongSORT as _StrongSORT
from yolo_tracker.trackers.multi_tracker_zoo import create_tracker
from yolo_tracker.yolov5.utils.torch_utils import select_device

from ...cache import cache
from ..decode_frame.decode_frame import DecodeFrame
from ..detection_2d.detection_2d import Detection2D
from ..detection_estimation import DetectionEstimation
from .tracking_2d import Tracking2D, Tracking2DResult

if TYPE_CHECKING:
    from ...payload import Payload


FILE = Path(__file__).resolve()
APPERCEPTION = FILE.parent.parent.parent.parent
WEIGHTS = APPERCEPTION / "weights"
reid_weights = WEIGHTS / "osnet_x0_25_msmt17.pt"


class StrongSORT(Tracking2D):
    def __init__(self, cache: "bool" = True) -> None:
        super().__init__()
        self.cache: "bool" = cache
        # self.ss_benchmarks: "list[list[list[float]]]" = []

    @cache
    def _run(self, payload: "Payload"):
        detections = Detection2D.get(payload)
        assert detections is not None

        sample_plans = DetectionEstimation.get(payload)

        images = DecodeFrame.get(payload)
        assert images is not None
        metadata: "List[Dict[int, Tracking2DResult]]" = []
        trajectories: "Dict[int, List[Tracking2DResult]]" = {}
        device = select_device("")
        strongsort = create_tracker('strongsort', reid_weights, device, False)
        assert isinstance(strongsort, _StrongSORT)
        curr_frame, prev_frame = None, None
        miss_track_count = 0
        with torch.no_grad():
            # ss_benchmark: "list[list[float]]" = []
            if hasattr(strongsort, 'model'):
                if hasattr(strongsort.model, 'warmup'):
                    strongsort.model.warmup()

            assert len(detections) == len(images) == len(sample_plans)
            # for idx, ((det, names, dids), im0s) in tqdm(enumerate(zip(detections, images)), total=len(images)):
            for idx, ((det, names, dids), im0s, sample_plan) in enumerate(zip(detections, images, sample_plans)):
                if not payload.keep[idx] or len(det) == 0:
                    metadata.append({})
                    strongsort.increment_ages()
                    prev_frame = im0s.copy()
                    continue

                # frame_benchmark: "list[float]" = []
                # frame_benchmark.append(time.time())
                im0 = im0s.copy()
                curr_frame = im0
                # frame_benchmark.append(time.time())

                if hasattr(strongsort, 'tracker') and hasattr(strongsort.tracker, 'camera_update'):
                    if prev_frame is not None and curr_frame is not None:
                        strongsort.tracker.camera_update(prev_frame, curr_frame, cache=self.cache)
                # frame_benchmark.append(time.time())

                confs = det[:, 4]
                output_ = strongsort.update(det.cpu(), im0)
                target_did = None
                if sample_plan and sample_plan.action:
                    target_did = sample_plan.action.target_obj_id

                # frame_benchmark.extend(_t)
                # frame_benchmark.append(time.time())

                if len(output_) > 0:
                    labels: "Dict[int, Tracking2DResult]" = {}
                    for output, conf, did in zip(output_, confs, dids):
                        obj_id = int(output[4])
                        cls = int(output[5])

                        bbox_left = output[0]
                        bbox_top = output[1]
                        bbox_w = output[2] - output[0]
                        bbox_h = output[3] - output[1]
                        labels[obj_id] = Tracking2DResult(
                            idx,
                            did,
                            obj_id,
                            bbox_left,
                            bbox_top,
                            bbox_w,
                            bbox_h,
                            names[cls],
                            conf.item(),
                        )
                        # if target_did and did == target_did:
                        #     print("found target object", target_did)
                        #     print("obj_id found:", obj_id in trajectories)
                        if obj_id not in trajectories and target_did and did == target_did:
                            miss_track_count += 1
                        if obj_id not in trajectories:
                            trajectories[obj_id] = []
                        trajectories[obj_id].append(labels[obj_id])
                    metadata.append(labels)
                else:
                    metadata.append({})

                # frame_benchmark.append(time.time())
                prev_frame = curr_frame

            #     ss = [
            #         t2 - t1
            #         for t1, t2
            #         in zip(frame_benchmark[:-1], frame_benchmark[1:])
            #     ]
            #     # print('[', ', '.join(map(str, ss)), '],')
            #     ss_benchmark.append(ss)

            # # for s in ss_benchmark:
            # #     print(s)
            # self.ss_benchmarks.append(ss_benchmark)
        print("miss track count:", miss_track_count)
        with open("./outputs/miss_count.txt", "r") as f:
            total_miss_count = int(f.read().strip())
            total_miss_count += miss_track_count
        with open("./outputs/miss_count.txt", "w") as f:
            f.write(str(total_miss_count))
        for trajectory in trajectories.values():
            for before, after in zip(trajectory[:-1], trajectory[1:]):
                before.next = after
                after.prev = before

        return None, {self.classname(): metadata}
