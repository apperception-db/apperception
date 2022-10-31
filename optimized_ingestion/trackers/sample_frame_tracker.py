import os

# limit the number of cpus used by high performance libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


import glob
import sys
from pathlib import Path

import torch

FILE = Path(__file__).resolve()
APPERCEPTION = FILE.parent.parent.parent
ROOT = APPERCEPTION / "submodules/Yolov5_StrongSORT_OSNet"  # yolov5 strongsort root directory
WEIGHTS = APPERCEPTION / "weights"

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / "yolov5") not in sys.path:
    sys.path.append(str(ROOT / "yolov5"))  # add yolov5 ROOT to PATH
if str(ROOT / "strong_sort") not in sys.path:
    sys.path.append(str(ROOT / "strong_sort"))  # add strong_sort ROOT to PATH
if str(ROOT / "strong_sort/deep/reid") not in sys.path:
    sys.path.append(str(ROOT / "strong_sort/deep/reid"))  # add strong_sort ROOT to PATH

import logging

from Yolov5_StrongSORT_OSNet.strong_sort.strong_sort import StrongSORT
from Yolov5_StrongSORT_OSNet.strong_sort.utils.parser import get_config
from Yolov5_StrongSORT_OSNet.yolov5.models.common import DetectMultiBackend
from Yolov5_StrongSORT_OSNet.yolov5.utils.dataloaders import (VID_FORMATS,
                                                              LoadImages)
from Yolov5_StrongSORT_OSNet.yolov5.utils.general import (LOGGER,
                                                          check_img_size,
                                                          colorstr, cv2,
                                                          increment_path,
                                                          non_max_suppression,
                                                          scale_coords,
                                                          strip_optimizer,
                                                          xyxy2xywh)
from Yolov5_StrongSORT_OSNet.yolov5.utils.plots import (Annotator, colors,
                                                        save_one_box)
from Yolov5_StrongSORT_OSNet.yolov5.utils.torch_utils import (select_device,
                                                              time_sync)

# remove duplicated stream handler to avoid duplicated logging
logging.getLogger().removeHandler(logging.getLogger().handlers[0])

yolo_weights = WEIGHTS / "yolov5s.pt"
config_strongsort = ROOT / "strong_sort/configs/strong_sort.yaml"
print(config_strongsort)
strong_sort_weights = WEIGHTS / "osnet_x0_25_msmt17.pt"  # model.pt path
save_txt = True
exist_ok = False

project = (APPERCEPTION / "runs/track",)  # save results to project/name
save_dir = str(APPERCEPTION / "tracks/") + "/"

# Load model
device = select_device("")
half = False
webcam = False
model = DetectMultiBackend(yolo_weights, device=device, dnn=False, data=None, fp16=half)
stride, names, pt = model.stride, model.names, model.pt
imgsz = check_img_size((640, 640), s=stride)  # check image size


@torch.no_grad()
def run(
    source=[],  # take in a list of frames, treat it as a video
    conf_thres=0.25,  # confidence threshold
    iou_thres=0.45,  # NMS IOU threshold
    max_det=1000,  # maximum detections per image
    show_vid=False,  # show results
    save_conf=False,  # save confidences in --save-txt labels
    save_crop=False,  # save cropped prediction boxes
    save_vid=False,  # save confidences in --save-txt labels
    nosave=False,  # do not save images/videos
    classes=None,  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False,  # class-agnostic NMS
    augment=False,  # augmented inference
    visualize=False,  # visualize features
    update=False,  # update all models
    line_thickness=3,  # bounding box thickness (pixels)
    hide_labels=False,  # hide labels
    hide_conf=False,  # hide confidences
    hide_class=False,  # hide IDs
):
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
    nr_sources = 1
    vid_path, vid_writer, txt_path = [None] * nr_sources, [None] * nr_sources, [None] * nr_sources

    # initialize StrongSORT
    cfg = get_config()
    cfg.merge_from_file(config_strongsort)

    # Create as many strong sort instances as there are video sources
    strongsort_list = []
    for i in range(nr_sources):
        strongsort_list.append(
            StrongSORT(
                strong_sort_weights,
                device,
                half,
                max_dist=cfg.STRONGSORT.MAX_DIST,
                max_iou_distance=cfg.STRONGSORT.MAX_IOU_DISTANCE,
                max_age=cfg.STRONGSORT.MAX_AGE,
                n_init=cfg.STRONGSORT.N_INIT,
                nn_budget=cfg.STRONGSORT.NN_BUDGET,
                mc_lambda=cfg.STRONGSORT.MC_LAMBDA,
                ema_alpha=cfg.STRONGSORT.EMA_ALPHA,
            )
        )
        strongsort_list[i].model.warmup()
    outputs = [None] * nr_sources

    # Run tracking
    model.warmup(imgsz=(1 if pt else nr_sources, 3, *imgsz))  # warmup
    dt, seen = [0.0, 0.0, 0.0, 0.0], 0
    curr_frames, prev_frames = [None] * nr_sources, [None] * nr_sources
    for frame_idx, (path, im, im0s, vid_cap, s) in enumerate(dataset):
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = (
            increment_path(save_dir / Path(path[0]).stem, mkdir=True) if visualize else False
        )
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # Apply NMS
        pred = non_max_suppression(
            pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det
        )
        dt[2] += time_sync() - t3

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            seen += 1
            if webcam:  # nr_sources >= 1
                p, im0, _ = path[i], im0s[i].copy(), dataset.count
                p = Path(p)  # to Path
                s += f"{i}: "
                txt_file_name = p.name
                save_path = save_dir + p.name  # im.jpg, vid.mp4, ...
            else:
                p, im0, _ = path, im0s.copy(), getattr(dataset, "frame", 0)
                p = Path(p)  # to Path
                # video file
                if isinstance(source, str) and source.endswith(VID_FORMATS):
                    txt_file_name = p.stem
                    save_path = save_dir + p.name  # im.jpg, vid.mp4, ...
                # folder with imgs
                else:
                    txt_file_name = p.parent.name  # get folder name containing current img
                    save_path = save_dir + p.parent.name  # im.jpg, vid.mp4, ...
            curr_frames[i] = im0

            txt_path = save_dir + txt_file_name  # im.txt
            s += "%gx%g " % im.shape[2:]  # print string
            imc = im0.copy() if save_crop else im0  # for save_crop

            annotator = Annotator(im0, line_width=2, pil=not ascii)
            if cfg.STRONGSORT.ECC:  # camera motion compensation
                strongsort_list[i].tracker.camera_update(prev_frames[i], curr_frames[i])

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                xywhs = xyxy2xywh(det[:, 0:4])
                confs = det[:, 4]
                clss = det[:, 5]

                # pass detections to strongsort
                t4 = time_sync()
                outputs[i] = strongsort_list[i].update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
                t5 = time_sync()
                dt[3] += t5 - t4

                # draw boxes for visualization
                if len(outputs[i]) > 0:
                    for j, (output, conf) in enumerate(zip(outputs[i], confs)):

                        bboxes = output[0:4]
                        id = output[4]
                        cls = output[5]
                        c = int(cls)
                        if save_txt:
                            # to MOT format
                            bbox_left = output[0]
                            bbox_top = output[1]
                            bbox_w = output[2] - output[0]
                            bbox_h = output[3] - output[1]
                            # Write MOT compliant results to file
                            with open(txt_path + ".txt", "a") as f:
                                # MOT format
                                f.write(
                                    ("%g," * 10 + f"{names[c]}\n")
                                    % (
                                        frame_idx,
                                        id,
                                        bbox_left,
                                        bbox_top,
                                        bbox_w,
                                        bbox_h,
                                        -1,
                                        -1,
                                        -1,
                                        i,
                                    )
                                )

                        if save_vid or save_crop or show_vid:  # Add bbox to image
                            c = int(cls)  # integer class
                            id = int(id)  # integer id
                            label = (
                                None
                                if hide_labels
                                else (
                                    f"{id} {names[c]}"
                                    if hide_conf
                                    else (
                                        f"{id} {conf:.2f}"
                                        if hide_class
                                        else f"{id} {names[c]} {conf:.2f}"
                                    )
                                )
                            )
                            annotator.box_label(bboxes, label, color=colors(c, True))
                            if save_crop:
                                txt_file_name = (
                                    txt_file_name
                                    if (isinstance(path, list) and len(path) > 1)
                                    else ""
                                )
                                save_one_box(
                                    bboxes,
                                    imc,
                                    file=save_dir
                                    / "crops"
                                    / txt_file_name
                                    / names[c]
                                    / f"{id}"
                                    / f"{p.stem}.jpg",
                                    BGR=True,
                                )

                LOGGER.info(f"{s}Done. YOLO:({t3 - t2:.3f}s), StrongSORT:({t5 - t4:.3f}s)")

            else:
                strongsort_list[i].increment_ages()
                LOGGER.info("No detections")

            # Stream results
            im0 = annotator.result()
            if show_vid:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_vid:
                if vid_path[i] != save_path:  # new video
                    vid_path[i] = save_path
                    if isinstance(vid_writer[i], cv2.VideoWriter):
                        vid_writer[i].release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                    save_path = str(
                        Path(save_path).with_suffix(".mp4")
                    )  # force *.mp4 suffix on results videos
                    vid_writer[i] = cv2.VideoWriter(
                        save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
                    )
                vid_writer[i].write(im0)

            prev_frames[i] = curr_frames[i]

    # Print results
    t = tuple(x / seen * 1e3 for x in dt)  # speeds per image
    LOGGER.info(
        f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms strong sort update per image at shape {(1, 3, *imgsz)}"
        % t
    )
    if save_txt or save_vid:
        s = (
            f"\n{len(list(glob.glob(save_dir + '/*.txt')))} tracks saved to {save_dir}"
            if save_txt
            else ""
        )
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(yolo_weights)  # update model (to fix SourceChangeWarning)