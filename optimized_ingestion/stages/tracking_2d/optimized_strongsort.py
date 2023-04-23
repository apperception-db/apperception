import cv2
import time
import torch
from pathlib import Path
from tqdm import tqdm
from typing import TYPE_CHECKING, Dict, List
from yolo_tracker.trackers.multi_tracker_zoo import StrongSORT as _StrongSORT
from yolo_tracker.trackers.multi_tracker_zoo import create_tracker
from yolo_tracker.yolov5.utils.torch_utils import select_device

from ...types import DetectionId
from ..decode_frame.decode_frame import DecodeFrame
from ..detection_2d.detection_2d import Detection2D
from .tracking_2d import Tracking2D, Tracking2DResult

if TYPE_CHECKING:

    from ...payload import Payload


FILE = Path(__file__).resolve()
APPERCEPTION = FILE.parent.parent.parent.parent
WEIGHTS = APPERCEPTION / "weights"
reid_weights = WEIGHTS / "osnet_x0_25_msmt17.pt"


def save_test_images(images, detections, output_path):
    for idx, image in enumerate(images):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        clss = detections[idx][1]
        obj_id = 0
        for detection in detections[idx][0]:
            x1, y1, x2, y2, cnf, cls = detection.cpu().numpy().astype(int)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, clss[cls] + '_' + str(obj_id), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            obj_id += 1
        cv2.imwrite(output_path + "/image_" + str(idx) + ".jpg", image)


def precompute_matching(tracked_obj_detection_info, current_detection_info):
    return False


def prematch_detection(last_frame_tracked, current_detection_infos):
    matched_output = []
    matched_confs = []
    for obj_id in last_frame_tracked:
        for i in range(len(current_detection_infos)):
            detection = current_detection_infos[i]
            if obj_id == i + 1 and precompute_matching(last_frame_tracked[obj_id], detection):
                matched_output.append(detection[:4] + [obj_id, detection[5]])
                matched_confs.append(detection[4])
                current_detection_infos.pop(i)
                break
    unmatched_output = current_detection_infos
    return matched_output, matched_confs, unmatched_output


class OptimizedStrongSORT(Tracking2D):
    def _run(self, payload: "Payload"):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        tracking_video = cv2.VideoWriter('test_tracking.avi', fourcc, 1, (1600, 900))
        start = time.time()
        total_prematch_time = 0
        total_img_process_time = 0
        # detection_info = DetectionEstimation.get(payload)
        # assert detection_info is not None

        detections = Detection2D.get(payload)
        assert detections is not None
        # print("original detections", detections)
        last_8_detections = detections[len(detections) * 97 // 99:]
        # matched_detections, unmatched_detections = prematch_detection(last_8_detections)

        images = DecodeFrame.get(payload)
        assert images is not None
        last_8_images = images[len(images) * 97 // 99:]
        # save_test_images(last_5s_images, last_5s_detections, "/data/apperception/apperception/optimized_sort_test_images")
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
                        img_process_time = time.time()
                        strongsort.tracker.camera_update(prev_frame, curr_frame, cache=True)
                        total_img_process_time += time.time() - img_process_time

                # prematch_start = time.time()
                # current_detections = det.cpu().numpy().tolist()
                # if len(metadata) > 0:
                #     last_frame_result = metadata[-1]
                #     matched_output, matched_confs, unmatched_detections = prematch_detection(last_frame_result, current_detections)
                # else:
                #     matched_output, matched_confs = [], []
                #     unmatched_detections = current_detections

                # matched_confs = torch.as_tensor(matched_confs)
                # if len(unmatched_detections) == 0:
                #     output_ = matched_output
                #     confs = matched_confs
                #     prematch_end = time.time()
                # else:
                #     unmatched_detections = torch.as_tensor(unmatched_detections)
                #     unmatched_confs = unmatched_detections[:, 4]
                #     # print("number of unmatched detections", len(unmatched_detections))
                #     # current_time = time.time() - start
                #     # print("current time", current_time)
                #     prematch_end = time.time()
                #     strongsort_output = np.array(strongsort.update(unmatched_detections, im0)).tolist()
                #     output_ = matched_output + strongsort_output
                #     confs = torch.cat((matched_confs, unmatched_confs))
                # total_prematch_time += prematch_end - prematch_start
                ### NO optimization
                # det = det[:len(det)//3]
                # print("number of detections", len(det))
                # current_time = time.time() - start
                # print("current time", current_time)
                confs = det[:, 4]
                output_ = strongsort.update(det.cpu(), im0)

                if len(output_) > 0:
                    labels: "Dict[int, Tracking2DResult]" = {}
                    # reconcile matched and unmatched
                    for i, (output, conf) in enumerate(zip(output_, confs)):
                        obj_id = int(output[4])
                        cls = int(output[5])

                        bbox_left = output[0]
                        bbox_top = output[1]
                        bbox_w = output[2] - output[0]
                        bbox_h = output[3] - output[1]
                        # write to video
                        cv2.rectangle(curr_frame, (int(bbox_left), int(bbox_top)), (int(bbox_left + bbox_w), int(bbox_top + bbox_h)), (0, 255, 0), 2)
                        cv2.putText(curr_frame, str(obj_id) + '_' + names[cls], (int(bbox_left), int(bbox_top)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                        labels[obj_id] = Tracking2DResult(
                            idx,
                            DetectionId(idx, i),
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
                    tracking_video.write(curr_frame)
                    metadata.append(labels)
                else:
                    metadata.append({})
                # print(idx, "\ncurrent metadata\n", metadata, "\n")
                prev_frame = curr_frame
            tracking_video.release()
        for trajectory in trajectories.values():
            for before, after in zip(trajectory[:-1], trajectory[1:]):
                before.next = after
                after.prev = before
        end = time.time()
        print("total time taken", end - start)
        print("total prematch time", total_prematch_time)
        print("total img process time", total_img_process_time)
        return None, {self.classname(): metadata}
