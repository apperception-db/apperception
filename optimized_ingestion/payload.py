from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Set

import cv2
import numpy as np
import numpy.typing as npt
from bitarray import bitarray
from tqdm import tqdm
from Yolov5_StrongSORT_OSNet.yolov5.utils.plots import Annotator, colors

from optimized_ingestion.utils.iterate_video import iterate_video

from .stages.depth_estimation import DepthEstimation
from .stages.filter_car_facing_sideway import FilterCarFacingSideway
from .stages.tracking_2d import Tracking2D

if TYPE_CHECKING:
    from .stages.stage import Stage
    from .trackers.yolov5_strongsort_osnet_tracker import TrackingResult
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
        keep, metadata = filter(self)

        if keep is None:
            keep = self.keep
        assert len(keep) == len(self.video), f"keep: {len(keep)}, video: {len(self.video)}"
        keep = keep & self.keep

        if metadata is None or len(metadata) == 0:
            metadata = self.metadata
        assert len(metadata) == 0 or metadata_len(metadata) == len(keep), f"metadata: {metadata_len(metadata)}, keep: {len(keep)}"
        metadata = {**self.metadata, **metadata}

        print(f"  filtered frames: {sum(keep) * 100.0 / len(keep)}%")
        print("\n".join(_split_keep(keep)))

        return Payload(self.video, keep, metadata)

    def save(self, filename: str, bbox: bool = True, depth: bool = True) -> None:
        video = cv2.VideoCapture(self.video.videofile)
        images = []
        idx = 0

        print("Annotating Video")
        trackings: "List[Dict[float, TrackingResult] | None]" = Tracking2D.get(self.metadata)
        depth_estimations: "List[npt.NDArray | None]" = DepthEstimation.get(self.metadata)
        filtered_objs: "List[Set[float] | None]" = FilterCarFacingSideway.get(self.metadata)
        frames = iterate_video(video)
        assert len(self.keep) == len(frames)
        assert len(self.keep) == len(trackings)
        assert len(self.keep) == len(depth_estimations)
        assert len(self.keep) == len(filtered_objs)
        for frame, k, tracking, depth_estimation, filtered_obj in tqdm(zip(iterate_video(video), self.keep, trackings, depth_estimations, filtered_objs), total=len(self.keep)):
            if not k:
                frame[:, :, 2] = 255

            if bbox and self.metadata is not None:
                filtered_obj = filtered_obj or set()
                if tracking is not None:
                    annotator = Annotator(frame, line_width=2)
                    for id, t in tracking.items():
                        if id not in filtered_obj:
                            continue
                        c = t.object_type
                        id = int(id)
                        x = int(t.bbox_left + t.bbox_w / 2)
                        y = int(t.bbox_top + t.bbox_h / 2)
                        label = f"{id} {c} conf: {t.confidence:.2f} dist: {depth_estimation[y, x]: .4f}"
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

        print("Saving depth")
        _filename = filename.split(".")
        _filename[-2] += "_depth"
        out = cv2.VideoWriter(
            ".".join(_filename),
            cv2.VideoWriter_fourcc(*"mp4v"),
            int(self.video.fps),
            (width, height),
        )
        blank = np.zeros((1600, 900, 3), dtype=np.uint8)
        if depth and depth_estimations is not None:
            for depth_estimation in tqdm(depth_estimations):
                if depth_estimation is None:
                    out.write(blank)
                    continue
                depth_estimation = depth_estimation[:, :, np.newaxis]
                _min = np.min(depth_estimation)
                _max = np.max(depth_estimation)
                depth_estimation = (depth_estimation - _min) * 256 / _max
                depth_estimation = depth_estimation.astype(np.uint8)
                depth_estimation = np.concatenate((depth_estimation, depth_estimation, depth_estimation), axis=2)
                out.write(depth_estimation)
        out.release()
        cv2.destroyAllWindows()


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