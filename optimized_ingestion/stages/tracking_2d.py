import pickle
import os
from typing import Dict, List, Optional, Tuple

from bitarray import bitarray

from ..payload import Payload
from ..trackers import yolov5_strongsort_osnet_tracker as tracker
from .stage import Stage


class Tracking2D(Stage):
    def __call__(self, payload: "Payload") -> "Tuple[Optional[bitarray], Optional[list]]":
        if os.path.exists('./_Tracking2D.pickle'):
            with open('./_Tracking2D.pickle', "rb") as f:
                return None, pickle.load(f)

        results = tracker.track(payload)
        results = sorted(results, key=lambda r: r.frame_idx)
        metadata: "List[dict | None]" = [None for _ in range(len(payload.video))]
        trajectories: "Dict[float, List[tracker.TrackingResult]]" = {}

        for row in results:
            idx = row.frame_idx

            if metadata[idx] is None:
                metadata[idx] = {}
            if "trackings" not in metadata[idx]:
                metadata[idx][Tracking2D.classname()] = {}
            Tracking2D.get(metadata[idx])[row.object_id] = row

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

        # with open('./_Tracking2D.pickle', "wb") as f:
        #     pickle.dump(metadata, f)

        return None, metadata
