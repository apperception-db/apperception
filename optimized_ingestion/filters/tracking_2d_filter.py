from typing import Dict, List, Optional, Tuple

from bitarray import bitarray

from ..payload import Payload
from ..trackers import yolov5_strongsort_osnet_tracker as tracker
from .filter import Filter


class Tracking2DFilter(Filter):
    def __call__(self, payload: "Payload") -> "Tuple[Optional[bitarray], Optional[list]]":
        _results = tracker.track(payload)

        _results = sorted(_results, key=lambda r: r.frame_idx)
        metadata: "List[dict]" = [None for _ in range(len(payload.frames))]
        trajectories: "Dict[float, List[tracker.TrackingResult]]" = {}

        for row in _results:
            idx = row.frame_idx

            if metadata[idx] is None:
                metadata[idx] = {}
            if "trackings" not in metadata[idx]:
                metadata[idx][Tracking2DFilter.classname()] = []
            Tracking2DFilter.get(metadata[idx]).append(row)

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

        return None, metadata
