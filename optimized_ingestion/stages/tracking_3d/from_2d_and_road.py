from typing import TYPE_CHECKING, Dict, List

import numpy as np
from pyquaternion import Quaternion

from ...payload import Payload
from ..tracking_2d import Tracking2D
from ..utils.is_annotated import is_annotated
from .tracking_3d import Tracking3D, Tracking3DResult

if TYPE_CHECKING:
    from ...trackers.yolov5_strongsort_osnet_tracker import TrackingResult
    from ..stage import StageOutput


class From2DAndRoad(Tracking3D):
    def __call__(self, payload: "Payload") -> "StageOutput":
        if not is_annotated(Tracking2D, payload):
            # payload = payload.filter(Tracking2D())
            raise Exception()

        metadata: "List[Dict[float, Tracking3DResult] | None]" = []
        trajectories: "Dict[float, List[Tracking3DResult]]" = {}
        video = payload.video
        trackings: "List[Dict[float, TrackingResult] | None]" = Tracking2D.get(payload.metadata)
        for i, (k, tracking, frame) in enumerate(zip(payload.keep, trackings, video)):
            if not k or tracking is None:
                metadata.append(None)
                continue
            
            if len(tracking) == 0:
                metadata.append({})
                continue

            trackings3d: "Dict[float, Tracking3DResult]" = {}
            [[fx, _, x0], [_, fy, y0], [_, _, s]] = frame.camera_intrinsic
            rotation = Quaternion(frame.camera_rotation).unit
            translation = np.array(frame.camera_translation)

            for object_id, t in tracking.items():
                y = int(t.bbox_top + t.bbox_h)
                Xs = []
                Ys = []
                Zs = []
                for x in range(int(t.bbox_left), int(t.bbox_left + t.bbox_w) + 1):
                    direction = np.array([(s * x + 0.5 - x0) / fx, (s * y + 0.5 - y0) / fy, 1])
                    direction = rotation.rotate(direction)

                    t = -translation[2] / direction[2]
                    Z = 0
                    X = direction[0] * t + translation[0]
                    Y = direction[1] * t + translation[1]

                    Xs.append(X)
                    Ys.append(Y)
                    Zs.append(Z)
                xyz = np.mean(np.stack((Xs, Ys, Zs)), axis=1)
                trackings3d [object_id] = Tracking3DResult(object_id, rotation.inverse.rotate(xyz - translation), xyz)
                if object_id not in trajectories:
                    trajectories[object_id] = []
                trajectories[object_id].append(trackings3d[object_id])
            metadata.append(trackings3d)

        for trajectory in trajectories.values():
            last = len(trajectory) - 1
            for i, t in enumerate(trajectory):
                if i > 0:
                    t.prev = trajectory[i - 1]
                if i < last:
                    t.next = trajectory[i + 1]

        return None, {self.classname(): metadata}
