import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import cv2
import numpy as np
import numpy.typing as npt
from tqdm import tqdm

# limit the number of cpus used by high performance libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


from pathlib import Path

import torch

if TYPE_CHECKING:
    from ..payload import Payload

FILE = Path(__file__).resolve()
APPERCEPTION = FILE.parent.parent.parent
WEIGHTS = APPERCEPTION / "weights"

import logging

from yolo_tracker.yolov5.models.common import DetectMultiBackend
from yolo_tracker.yolov5.utils.dataloaders import LoadImages
from yolo_tracker.yolov5.utils.general import check_img_size, non_max_suppression, scale_coords
from yolo_tracker.yolov5.utils.torch_utils import select_device
from yolo_tracker.trackers.multi_tracker_zoo import create_tracker

# remove duplicated stream handler to avoid duplicated logging
logging.getLogger().removeHandler(logging.getLogger().handlers[0])

yolo_weights = WEIGHTS / "yolov5s.pt"
reid_weights = WEIGHTS / "osnet_x0_25_msmt17.pt"  # model.pt path


# Load model
device = select_device("")
half = False
model = DetectMultiBackend(yolo_weights, device=device, dnn=False, data=None, fp16=half)
stride, names, pt = model.stride, model.names, model.pt
imgsz = check_img_size((640, 640), s=stride)  # check image size


@torch.no_grad()
def track(
    source: "Payload",  # take in a list of frames, treat it as a video
    conf_thres=0.25,  # confidence threshold
    iou_thres=0.45,  # NMS IOU threshold
    max_det=1000,  # maximum detections per image
    classes=None,  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False,  # class-agnostic NMS
    augment=False,  # augmented inference
):
    video = cv2.VideoCapture(source.video.videofile)
    video_len = video.get(cv2.CAP_PROP_FRAME_COUNT)
    video.release()
    cv2.destroyAllWindows()
    dataset = LoadImages(source.video.videofile, img_size=imgsz, stride=stride, auto=pt)

    nr_sources = 1
    labels: "List[TrackingResult]" = []

    # Create a strong sort instances as there is an only one video source
    tracker = create_tracker('strongsort', reid_weights, device, half)
    if hasattr(tracker, 'model'):
        if hasattr(tracker.model, 'warmup'):
            tracker.model.warmup()

    # Run tracking
    model.warmup(imgsz=(1 if pt else nr_sources, 3, *imgsz))  # warmup
    dt, seen = [0.0, 0.0, 0.0, 0.0], 0
    curr_frame, prev_frame = None, None
    for frame_idx, (path, im, im0s, vid_cap, s) in tqdm(enumerate(dataset), total=video_len):
        if not source.keep[frame_idx]:
            continue
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        pred = model(im, augment=augment, visualize=False)
        t3 = time_sync()
        dt[1] += t3 - t2

        # Apply NMS
        pred = non_max_suppression(
            pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det
        )
        dt[2] += time_sync() - t3

        # Process detections
        assert isinstance(pred, list)
        assert len(pred) == 1
        det = pred[0]
        seen += 1
        p, im0, _ = path, im0s.copy(), getattr(dataset, "frame", 0)
        p = Path(p)  # to Path
        curr_frame = im0

        s += "%gx%g " % im.shape[2:]  # print string

        if hasattr(tracker, 'tracker') and hasattr(tracker.tracker, 'camera_update'):
            if prev_frame is not None and curr_frame is not None: # camera motion compensation
                tracker.tracker.camera_update(prev_frame, curr_frame)

        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()  # xyxy

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

            confs = det[:, 4]

            # pass detections to strongsort
            t4 = time_sync()
            output_ = tracker.update(det.cpu(), im0)
            t5 = time_sync()
            dt[3] += t5 - t4

            # draw boxes for visualization
            if len(output_) > 0:
                for output, conf in zip(output_, confs):

                    id = output[4]
                    cls = output[5]
                    c = int(cls)

                    # to MOT format
                    bbox_left = output[0]
                    bbox_top = output[1]
                    bbox_w = output[2] - output[0]
                    bbox_h = output[3] - output[1]
                    labels.append(
                        TrackingResult(
                            frame_idx,
                            int(id),
                            bbox_left,
                            bbox_top,
                            bbox_w,
                            bbox_h,
                            0,  # TODO: remove
                            f"{names[c]}",
                            conf.item(),
                        )
                    )

            # LOGGER.info(f"{s}Done. YOLO:({t3 - t2:.3f}s), StrongSORT:({t5 - t4:.3f}s)")

        else:
            tracker.increment_ages()
            # LOGGER.info("No detections")

        prev_frame = curr_frame

    # Print results
    # t = tuple(x / seen * 1e3 for x in dt)  # speeds per image
    # LOGGER.info(
    #     f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms strong sort update per image at shape {(1, 3, *imgsz)}"
    #     % t
    # )
    return labels


@dataclass
class TrackingResult:
    frame_idx: int
    object_id: float
    bbox_left: float
    bbox_top: float
    bbox_w: float
    bbox_h: float
    pred_idx: int
    object_type: str
    confidence: float
    prev: "TrackingResult | None" = None
    next: "TrackingResult | None" = None
