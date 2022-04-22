import os
import sys

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(CURRENT_DIR, "../yolov5-deepsort/yolov5/"))
sys.path.append(os.path.join(CURRENT_DIR, "../yolov5-deepsort/"))

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
from deep_sort_pytorch.deep_sort import DeepSort
from deep_sort_pytorch.utils.parser import get_config
from yolov5.models.experimental import attempt_load
from yolov5.utils.datasets import LoadImages
from yolov5.utils.downloads import attempt_download
from yolov5.utils.general import (check_img_size, non_max_suppression,
                                  scale_coords, xyxy2xywh)
from yolov5.utils.torch_utils import select_device

from ..types import BoundingBox, TrackedObject


@dataclass
class YoloV5Opt:
    source: str
    yolo_weights: str = os.path.join(CURRENT_DIR, "../yolov5-deepsort/yolov5/weights/yolov5s.pt")
    deep_sort_weights: str = os.path.join(
        CURRENT_DIR, "../yolov5-deepsort/deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7"
    )
    # output: str = 'inference/output'
    img_size: Union[Tuple[int, int], int] = 640
    conf_thres: float = 0.4
    iou_thres: float = 0.5
    # fourcc: str = 'mp4v'
    device: str = ""
    # show_vid: bool = False
    # save_vid: bool = False
    # save_txt: bool = False
    classes: Optional[List[int]] = None
    agnostic_nms: bool = False
    augment: bool = False
    # evaluate: bool = False
    config_deepsort: str = os.path.join(
        CURRENT_DIR, "../yolov5-deepsort/deep_sort_pytorch/configs/deep_sort.yaml"
    )


def detect(opt: YoloV5Opt):
    source, yolo_weights, deep_sort_weights, imgsz = (
        opt.source,
        opt.yolo_weights,
        opt.deep_sort_weights,
        opt.img_size,
    )

    crop = BoundingBox(0, 0, 100, 100)

    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    attempt_download(deep_sort_weights, repo="mikel-brostrom/Yolov5_DeepSort_Pytorch")
    deepsort = DeepSort(
        deep_sort_weights,
        max_dist=cfg.DEEPSORT.MAX_DIST,
        min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
        max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
        max_age=cfg.DEEPSORT.MAX_AGE,
        n_init=cfg.DEEPSORT.N_INIT,
        nn_budget=cfg.DEEPSORT.NN_BUDGET,
        use_cuda=True,
    )

    # Initialize
    device = select_device(opt.device)

    half = device.type != "cpu"  # half precision only supported on CUDA
    # Load model
    model = attempt_load(yolo_weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    names = model.module.names if hasattr(model, "module") else model.names  # get class names
    if half:
        model.half()  # to FP16

    dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, "module") else model.names

    # Run inference
    # if device.type != "cpu":
    #     _, img, _, _ = dataset[0]
    #     h, w = img.shape[1:]

    #     # crop image
    #     x1, y1, x2, y2 = [
    #         int(v / 100.0)
    #         for v in [
    #             w * crop.x1,
    #             h * crop.y1,
    #             w * crop.x2,
    #             h * crop.y2,
    #         ]
    #     ]

    #     img = img[:, y1:y2, x1:x2]
    #     model(
    #         torch.zeros(1, 3, img.shape[1], img.shape[2])
    #         .to(device)
    #         .type_as(next(model.parameters()))
    #     )  # run once

    formatted_result: Dict[str, TrackedObject] = {}
    for frame_idx, (_, img, im0s, _, _) in enumerate(dataset):
        h, w = img.shape[1:]

        # crop image
        x1, y1, x2, y2 = [
            int(v / 100.0)
            for v in [
                w * crop.x1,
                h * crop.y1,
                w * crop.x2,
                h * crop.y2,
            ]
        ]
        img = img[:, y1:y2, x1:x2]

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(
            pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms
        )

        # Process detections
        for det in pred:  # detections per image

            if det is None or not len(det):
                deepsort.increment_ages()
                continue

            # add padding from cropped frame
            det[:, :4] += torch.tensor([[x1, y1, x1, y1]]).to(device)

            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(
                (h, w),
                det[:, :4],
                im0s.shape,
            ).round()

            xywhs = xyxy2xywh(det[:, 0:4])
            confs = det[:, 4]
            clss = det[:, 5]

            # pass detections to deepsort
            outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0s)

            # collect result bounding boxes
            for output in outputs:

                y1, x1, y2, x2, id, c = [int(o) for o in output]
                bboxes = BoundingBox(x1, y1, x2, y2)
                item_id = f"{names[c]}-{str(id)}"

                if item_id not in formatted_result:
                    formatted_result[item_id] = TrackedObject(object_type=names[c])

                formatted_result[item_id].bboxes.append(bboxes)
                formatted_result[item_id].frame_num.append(frame_idx)

    return formatted_result


def yolov5_deepsort_video_track(opt: YoloV5Opt):
    with torch.no_grad():
        return detect(opt)
