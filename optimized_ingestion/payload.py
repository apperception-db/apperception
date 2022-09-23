from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

import cv2
import numpy as np
import numpy.typing as npt
from bitarray import bitarray
from Yolov5_StrongSORT_OSNet.yolov5.utils.plots import Annotator, colors

from optimized_ingestion.utils.iterate_video import iterate_video

from .stages.depth_estimation import DepthEstimation
from .stages.filter_car_facing_sideway import FilterCarFacingSideway
from .stages.tracking_2d import Tracking2D

if TYPE_CHECKING:
    from .stages.stage import Stage
    from .trackers.yolov5_strongsort_osnet_tracker import TrackingResult
    from .video import Video


# TODO: add Generic depending on the type of stage applied
@dataclass
class Payload:
    video: "Video"
    keep: "bitarray"
    metadata: "Optional[List[Any]]"

    def __init__(
        self,
        video: "Video",
        keep: "Optional[bitarray]" = None,
        metadata: "Optional[List[Any]]" = None,
    ):
        self.keep = _default_keep(video, keep)
        self.video = video
        if metadata is not None and len(metadata) != len(video):
            raise Exception(f"metadata: {len(metadata)}, video: {len(video)}")
        self.metadata = metadata

    def filter(self, filter: "Stage"):
        print("Stage: ", filter.classname())
        keep, metadata = filter(self)

        assert keep is None or len(keep) == len(self.video), f"keep: {len(keep)}, video: {len(self.video)}"
        if keep is None:
            keep = self.keep
        else:
            keep = keep & self.keep

        assert metadata is None or len(metadata) == len(keep), f"metadata: {len(metadata)}, keep: {len(keep)}"
        if metadata is None:
            metadata = self.metadata
        elif self.metadata is not None:
            metadata = [_merge(m1, m2) for m1, m2 in zip(self.metadata, metadata)]

        print(f"  filtered frames: {sum(keep) * 100.0 / len(keep)}%")
        print("\n".join(_split_keep(keep)))

        return Payload(self.video, self.keep & keep, metadata)

    def save(self, filename: str, bbox: bool = True, depth: bool = True) -> None:
        video = cv2.VideoCapture(self.video.videofile)
        n_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        images = []
        idx = 0
        # while video.isOpened():
        for frame in iterate_video(video):
            # ret, frame = video.read()
            # if not ret:
            #     break
            if not self.keep[idx]:
                frame[:, :, 2] = 255

            if bbox and self.metadata is not None:
                trackings: "Dict[float, TrackingResult] | None" = Tracking2D.get(self.metadata[idx])
                _depth: "npt.NDArray" = DepthEstimation.get(self.metadata[idx])
                filtered_obj: "Set[float]" = FilterCarFacingSideway.get(self.metadata[idx]) or set()
                if trackings is not None:
                    annotator = Annotator(frame, line_width=2)
                    for id, t in trackings.items():
                        if id not in filtered_obj:
                            continue
                        c = t.object_type
                        id = int(id)
                        x = int(t.bbox_left + t.bbox_w / 2)
                        y = int(t.bbox_top + t.bbox_h / 2)
                        label = f"{id} {c} conf: {t.confidence:.2f} dist: {_depth[y, x]: .4f}"
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
        # video.release()
        # cv2.destroyAllWindows()

        height, width, _ = images[0].shape
        out = cv2.VideoWriter(
            filename, cv2.VideoWriter_fourcc(*"mp4v"), int(self.video.fps), (width, height)
        )
        for image in images:
            out.write(image)
        out.release()
        cv2.destroyAllWindows()

        _filename = filename.split(".")
        _filename[-2] += "_depth"
        out = cv2.VideoWriter(
            ".".join(_filename),
            cv2.VideoWriter_fourcc(*"mp4v"),
            int(self.video.fps),
            (width, height),
        )
        blank = np.zeros((1600, 900, 3), dtype=np.uint8)
        if depth and self.metadata is not None:
            for m in self.metadata:
                _depth = DepthEstimation.get(m)
                if _depth is None:
                    out.write(blank)
                else:
                    _depth = _depth[:, :, np.newaxis]
                    _min = np.min(_depth)
                    _max = np.max(_depth)
                    _depth = (_depth - _min) * 256 / _max
                    _depth = _depth.astype(np.uint8)
                    _depth = np.concatenate((_depth, _depth, _depth), axis=2)
                    out.write(_depth)
        out.release()
        cv2.destroyAllWindows()


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
