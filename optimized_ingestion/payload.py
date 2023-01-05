import cv2
from bitarray import bitarray
from dataclasses import dataclass
from tqdm import tqdm
from typing import TYPE_CHECKING, Dict, List, Optional, Set
from yolo_tracker.yolov5.utils.plots import Annotator, colors

from .stages.filter_car_facing_sideway import FilterCarFacingSideway
from .stages.tracking_2d.tracking_2d import Tracking2D
from .stages.tracking_3d.tracking_3d import Tracking3D, Tracking3DResult
from .utils.iterate_video import iterate_video
from .video_skipped import VideoSkipped

if TYPE_CHECKING:
    from .stages.stage import Stage
    from .stages.tracking_2d.tracking_2d import Tracking2DResult
    from .video import Video


Metadata = Dict[str, list]


# TODO: add Generic depending on the type of stage applied
@dataclass
class Payload:
    video: "Video"
    keep: "bitarray"
    metadata: "Metadata"

    def __init__(
        self,
        video: "Video",
        keep: "Optional[bitarray]" = None,
        metadata: "Optional[Metadata]" = None,
    ):
        self.keep = _default_keep(video, keep)
        self.video = video
        if metadata is None:
            metadata = {}
        if not all(len(m) == len(video) for m in metadata.values()) :
            raise Exception(f"metadata: {len(metadata)}, video: {len(video)}")
        self.metadata = metadata

    def filter(self, filter: "Stage"):
        print("Stage: ", filter.classname())
        keep, metadata = filter.run(self)
        print(keep and len(keep))
        print(metadata and len(filter.get(metadata)))

        if keep is None:
            keep = self.keep
        assert len(keep) == len(self.video), f"keep: {len(keep)}, video: {len(self.video)}"
        keep = keep & self.keep

        if metadata is None or len(metadata) == 0:
            metadata = self.metadata

        if len(metadata) != 0:
            metadata_l = metadata_len(metadata)
            assert metadata_l == len(keep), f"metadata: {metadata_l}, keep: {len(keep)}"

        metadata = {**self.metadata, **metadata}

        print(f"  filtered frames: {sum(keep) * 100.0 / len(keep)}%")
        print("\n".join(_split_keep(keep, 100)))

        return Payload(self.video, keep, metadata)

    def save(self, filename: str, bbox: bool = True, depth: bool = True) -> None:
        video = cv2.VideoCapture(self.video.videofile)
        images = []
        idx = 0

        print("Annotating Video")
        trackings: "List[Dict[float, Tracking2DResult] | None]" = Tracking2D.get(self.metadata)
        trackings3d: "List[Dict[float, Tracking3DResult]]" = Tracking3D.get(self.metadata)
        filtered_objs: "List[Set[float] | None]" = FilterCarFacingSideway.get(self.metadata)
        frames = iterate_video(video)
        assert len(self.keep) == len(frames)
        assert len(self.keep) == len(trackings)
        assert len(self.keep) == len(filtered_objs)
        for frame, k, tracking, tracking3d, filtered_obj in tqdm(zip(iterate_video(video), self.keep, trackings, trackings3d, filtered_objs), total=len(self.keep)):
            if not k:
                frame[:, :, 2] = 255

            if bbox and self.metadata is not None:
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
            filename, cv2.VideoWriter_fourcc(*"mp4v"), int(self.video.fps), (width, height)
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
        #     int(self.video.fps),
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


def metadata_len(metadata: "Dict[str, list]") -> "int | None":
    length: "int | None" = None
    for v in metadata.values():
        if length is None:
            length = len(v)
        elif length != len(v):
            raise Exception()
    return length


def _merge(meta1, meta2):
    if not meta1:
        return meta2
    if not meta2:
        return meta1
    return {**meta1, **meta2}


def _default_keep(video: "Video", keep: "Optional[bitarray]" = None):
    if keep is None:
        if isinstance(video, VideoSkipped):
            keep = bitarray([c is not None for c in video.interpolated_frames])
        else:
            keep = bitarray(len(video))
            keep.setall(1)
    elif len(keep) != len(video):
        raise Exception()
    return keep


def _split_keep(keep: "bitarray", size: int = 64) -> "List[str]":
    out: "List[str]" = []
    i = 0
    while i * size < len(keep):
        curr = i * size
        next = min((i + 1) * size, len(keep))
        out.append("".join(["K" if k else "." for k in keep[curr:next]]))
        i += 1

    return out
