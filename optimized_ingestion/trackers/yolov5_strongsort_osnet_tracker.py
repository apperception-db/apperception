import json
import os
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, List, Tuple

import cv2
import numpy as np
import numpy.typing as npt
import torch
from tqdm import tqdm

# limit the number of cpus used by high performance libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


if TYPE_CHECKING:
    from ..payload import Payload

from yolo_tracker.trackers.multi_tracker_zoo import create_tracker
from yolo_tracker.yolov5.utils.augmentations import letterbox
from yolo_tracker.yolov5.utils.general import (check_img_size,
                                               non_max_suppression,
                                               scale_boxes)
from yolo_tracker.yolov5.utils.torch_utils import select_device  # , time_sync

from ..stages.decode_frame import DecodeFrame

FILE = Path(__file__).resolve()
APPERCEPTION = FILE.parent.parent.parent
WEIGHTS = APPERCEPTION / "weights"
reid_weights = WEIGHTS / "osnet_x0_25_msmt17.pt"  # model.pt path


# Load model
device = select_device("")
print("Using", device)
half = False
model = torch.hub.load('ultralytics/yolov5', 'yolov5s').model.to(device)
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
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)

    nr_sources = 1
    labels: "List[TrackingResult]" = []

    # Create a strong sort instances as there is an only one video source
    strongsort = create_tracker('strongsort', reid_weights, device, half)
    if hasattr(strongsort, 'model'):
        if hasattr(strongsort.model, 'warmup'):
            strongsort.model.warmup()

    # Run tracking
    model.eval()
    model.warmup(imgsz=(1 if pt else nr_sources, 3, *imgsz))  # warmup
    # dt, seen = [0.0, 0.0, 0.0, 0.0], 0
    curr_frame, prev_frame = None, None
    for frame_idx, im, im0s in tqdm(dataset):
        if not source.keep[frame_idx]:
            continue

        # t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        # t2 = time_sync()
        # dt[0] += t2 - t1

        # Inference
        pred = model(im, augment=augment, visualize=False)
        # t3 = time_sync()
        # dt[1] += t3 - t2

        # Apply NMS
        pred = non_max_suppression(
            pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det
        )
        # dt[2] += time_sync() - t3

        # Process detections
        assert isinstance(pred, list)
        assert len(pred) == 1
        det = pred[0]
        # seen += 1
        im0, _ = im0s.copy(), getattr(dataset, "frame", 0)
        curr_frame = im0

        # s += "%gx%g " % im.shape[2:]  # print string

        if hasattr(strongsort, 'tracker') and hasattr(strongsort.tracker, 'camera_update'):
            if prev_frame is not None and curr_frame is not None:  # camera motion compensation
                strongsort.tracker.camera_update(prev_frame, curr_frame)
        if det is not None and len(det):

            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # xyxy

            # Print results
            # for c in det[:, -1].unique():
            #     n = (det[:, -1] == c).sum()  # detections per class
            #     s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

            confs = det[:, 4]

            # pass detections to strongsort
            # t4 = time_sync()
            output_ = strongsort.update(det.cpu(), im0)
            # t5 = time_sync()
            # dt[3] += t5 - t4

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
            strongsort.increment_ages()
            # LOGGER.info("No detections")

        prev_frame = curr_frame

    # Print results
    # t = tuple(x / seen * 1e3 for x in dt)  # speeds per image
    # LOGGER.info(
    #     f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms strong sort update per image at shape {(1, 3, *imgsz)}"
    #     % t
    # )
    with open(f"strongsort_{source.video.videofile.split('/')[-1]}.json", "w") as f:
        json.dump(strongsort.benchmark, f)
    return labels


@dataclass
class TrackingResult:
    frame_idx: int
    object_id: int
    bbox_left: float
    bbox_top: float
    bbox_w: float
    bbox_h: float
    pred_idx: int
    object_type: str
    confidence: float
    prev: "TrackingResult | None" = None
    next: "TrackingResult | None" = None


# ImageOutput = Tuple[int, npt.NDArray, npt.NDArray, str]
ImageOutput = Tuple[int, npt.NDArray, npt.NDArray]


class LoadImages(Iterator[ImageOutput], Iterable[ImageOutput]):
    # YOLOv5 image/video dataloader, i.e. `python detect.py --source image.jpg/vid.mp4`
    def __init__(self, payload: "Payload", img_size=640, stride=32, auto=True, transforms=None, vid_stride=1):

        self.img_size = img_size
        self.stride = stride
        self.mode = 'image'
        self.auto = auto
        self.transforms = transforms  # optional
        self.vid_stride = vid_stride  # video frame-rate stride
        self._new_video(payload.video.videofile)  # new video
        self.keep = payload.keep

        images = DecodeFrame.get(payload)
        assert images is not None
        self.images = images

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self) -> "ImageOutput":
        if self.frame >= self.frames:
            raise StopIteration

        # Read video
        self.mode = 'video'
        im0 = self.images[self.frame]
        assert isinstance(im0, np.ndarray)

        frame_idx = self.frame
        self.frame += self.vid_stride
        # im0 = self._cv2_rotate(im0)  # for use if cv2 autorotation is False
        # s = f'video {self.count + 1}/{self.len} ({self.frame}/{self.frames}): '

        if self.transforms:
            im = self.transforms(im0)  # transforms
        else:
            im = letterbox(im0, self.img_size, stride=self.stride, auto=self.auto)[0]  # padded resize
            im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            im = np.ascontiguousarray(im)  # contiguous

        # return frame_idx, im, im0, s
        return frame_idx, im, im0

    def _new_video(self, path):
        # Create a new video capture object
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.len = int(self.frames / self.vid_stride)
        self.orientation = int(self.cap.get(cv2.CAP_PROP_ORIENTATION_META))  # rotation degrees
        # self.cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)  # disable https://github.com/ultralytics/yolov5/issues/8493

    def _cv2_rotate(self, im):
        # Rotate a cv2 video manually
        if self.orientation == 0:
            return cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)
        elif self.orientation == 180:
            return cv2.rotate(im, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif self.orientation == 90:
            return cv2.rotate(im, cv2.ROTATE_180)
        return im

    def __len__(self):
        return self.len
