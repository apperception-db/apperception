import cv2
from tqdm import tqdm
from yolo_tracker.yolov5.utils.plots import Annotator, colors

from ..payload import Payload
from ..stages.filter_car_facing_sideway import FilterCarFacingSideway
from ..stages.tracking_2d.tracking_2d import Tracking2D
from ..stages.tracking_3d.tracking_3d import Tracking3D
from ..utils.iterate_video import iterate_video


def save_video(payload: "Payload", filename: str, bbox: bool = True, depth: bool = True) -> None:
    video = cv2.VideoCapture(payload.video.videofile)
    images = []
    idx = 0

    print("Annotating Video")
    trackings = Tracking2D.get(payload.metadata)
    assert trackings is not None
    trackings3d = Tracking3D.get(payload.metadata)
    assert trackings3d is not None
    filtered_objs = FilterCarFacingSideway.get(payload.metadata)
    assert filtered_objs is not None

    frames = iterate_video(video)
    assert len(payload.keep) == len(frames)
    assert len(payload.keep) == len(trackings)
    assert len(payload.keep) == len(filtered_objs)
    for frame, k, tracking, tracking3d, filtered_obj in tqdm(zip(iterate_video(video), payload.keep, trackings, trackings3d, filtered_objs), total=len(payload.keep)):
        if not k:
            frame[:, :, 2] = 255

        if bbox and payload.metadata is not None:
            filtered_obj = filtered_obj or set()
            if tracking is not None:
                annotator = Annotator(frame, line_width=2)
                for id, t in tracking.items():
                    t3d = tracking3d[id]
                    if t3d.point_from_camera[2] >= 0:
                        continue
                    if id not in filtered_obj:
                        continue
                    c = t.object_type
                    id = int(id)
                    label = f"{id} {c} conf: {t.confidence:.2f} dist: {t3d.point_from_camera[2]: .4f}"
                    annotator.box_label(
                        [
                            t.bbox_left,
                            t.bbox_top,
                            t.bbox_left + t.bbox_w,
                            t.bbox_top + t.bbox_h,
                        ],
                        label,
                        color=colors(id, True),
                    )
                frame = annotator.result()

        images.append(frame)
        idx += 1

    print("Saving Video")
    height, width, _ = images[0].shape
    out = cv2.VideoWriter(
        filename, cv2.VideoWriter_fourcc(*"mp4v"), int(payload.video.fps), (width, height)
    )
    for image in tqdm(images):
        out.write(image)
    out.release()
    cv2.destroyAllWindows()

    # print("Saving depth")
    # _filename = filename.split(".")
    # _filename[-2] += "_depth"
    # out = cv2.VideoWriter(
    #     ".".join(_filename),
    #     cv2.VideoWriter_fourcc(*"mp4v"),
    #     int(payload.video.fps),
    #     (width, height),
    # )
    # blank = np.zeros((1600, 900, 3), dtype=np.uint8)
    # if depth and depth_estimations is not None:
    #     for depth_estimation in tqdm(depth_estimations):
    #         if depth_estimation is None:
    #             out.write(blank)
    #             continue
    #         depth_estimation = depth_estimation[:, :, np.newaxis]
    #         _min = np.min(depth_estimation)
    #         _max = np.max(depth_estimation)
    #         depth_estimation = (depth_estimation - _min) * 256 / _max
    #         depth_estimation = depth_estimation.astype(np.uint8)
    #         depth_estimation = np.concatenate((depth_estimation, depth_estimation, depth_estimation), axis=2)
    #         out.write(depth_estimation)
    # out.release()
    # cv2.destroyAllWindows()
