from typing import TYPE_CHECKING, Iterable, Iterator, NamedTuple, Tuple
import os

from tqdm import tqdm

# limit the number of cpus used by high performance libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import cv2
import torch
import numpy as np
import numpy.typing as npt
from yolo_tracker.yolov5.utils.torch_utils import select_device
from yolo_tracker.yolov5.utils.augmentations import letterbox
from yolo_tracker.yolov5.utils.general import (check_img_size,
                                               non_max_suppression,
                                               scale_boxes)

from .detection_2d import Detection2D
from ...stages.decode_frame import DecodeFrame

if TYPE_CHECKING:
    from yolo_tracker.yolov5.models.common import DetectMultiBackend

    from ...payload import Payload
    from ...stages.stage import StageOutput


class YoloDetection(Detection2D):
    def __init__(
        self,
        half: bool = False,
        conf_thres: float = 0.25, 
        iou_thres: float = 0.45,
        max_det: int = 1000,
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
    ):
        self.device = select_device("")
        self.model: "DetectMultiBackend" = torch.hub.load('ultralytics/yolov5', 'yolov5s').model.to(self.device)
        stride, names, pt = self.model.stride, self.model.names, self.model.pt
        self.imgsz = check_img_size((640, 640), s=stride)
        self.half = half
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        self.classes = classes
        self.agnostic_nms = agnostic_nms
        self.augment = augment

    def _run(self, payload: "Payload") -> "StageOutput":
        with torch.no_grad():
            dataset = LoadImages(payload, img_size=self.imgsz)
            self.model.eval()
            self.model.warmup(imgsz=(1, 3, *self.imgsz))  # warmup
            for frame_idx, im, im0s in tqdm(dataset):
                if not payload.keep[frame_idx]:
                    continue
                # t1 = time_sync()
                im = torch.from_numpy(im).to(self.device)
                im = im.half() if self.half else im.float()
                im /= 255.0  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim
                # t2 = time_sync()
                # dt[0] += t2 - t1

                # Inference
                pred = self.model(im, augment=self.augment)
                # t3 = time_sync()
                # dt[1] += t3 - t2

                # Apply NMS
                pred = non_max_suppression(
                    pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det
                )
                # dt[2] += time_sync() - t3

                # Process detections
                assert isinstance(pred, list)
                assert len(pred) == 1
                det = pred[0]
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0s.shape).round()
        return super()._run(payload)


class ImageOutput(NamedTuple):
    frame_idx: int
    image: "npt.NDArray"
    original_image: "npt.NDArray"
    # path: str


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
        return ImageOutput(frame_idx, im, im0)

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
