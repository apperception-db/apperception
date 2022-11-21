from bitarray import bitarray
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from ...trackers import yolov5_strongsort_osnet_tracker as tracker
from ..stage import Stage

if TYPE_CHECKING:
    from ...payload import Payload


class Tracking2D(Stage):
    def _run(self, payload: "Payload") -> "Tuple[Optional[bitarray], Optional[Dict[str, list]]]":
        # if os.path.exists("./_Tracking2D.pickle"):
        #     with open("./_Tracking2D.pickle", "rb") as f:
        #         return None, {self.classname(): pickle.load(f)}

        results = tracker.track(payload)
        results = sorted(results, key=lambda r: r.frame_idx)
        metadata: "List[Dict[float, tracker.TrackingResult] | None]" = []
        trajectories: "Dict[float, List[tracker.TrackingResult]]" = {}

        for k in payload.keep:
            if k:
                metadata.append({})
            else:
                metadata.append(None)

        for row in map(tracking_result_to_tracking_2d_result, results):
            idx = row.frame_idx
            metadata[idx][row.object_id] = row

            if row.object_id not in trajectories:
                trajectories[row.object_id] = []
            trajectories[row.object_id].append(row)

        for trajectory in trajectories.values():
            last = len(trajectory) - 1
            for i, t in enumerate(trajectory):
                if i > 0:
                    t.prev = trajectory[i - 1]
                if i < last:
                    t.next = trajectory[i + 1]

        # with open("./_Tracking2D.pickle", "wb") as f:
        #     pickle.dump(metadata, f)

        return None, {self.classname(): metadata}


def tracking_result_to_tracking_2d_result(t: "tracker.TrackingResult"):
    return Tracking2DResult(
        t.frame_idx,
        t.object_id,
        t.bbox_left,
        t.bbox_top,
        t.bbox_w,
        t.bbox_h,
        t.object_type,
        t.confidence
    )


@dataclass
class Tracking2DResult:
    frame_idx: int
    object_id: int
    bbox_left: float
    bbox_top: float
    bbox_w: float
    bbox_h: float
    object_type: str
    confidence: float
    next: "Tracking2DResult | None" = field(default=None, compare=False, repr=False)
    prev: "Tracking2DResult | None" = field(default=None, compare=False, repr=False)
