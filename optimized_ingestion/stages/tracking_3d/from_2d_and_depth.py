from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
from bitarray import bitarray
from pyquaternion import Quaternion

from ...utils.depth_to_3d import depth_to_3d
from ..depth_estimation import DepthEstimation
from ..tracking_2d import Tracking2D
from ..utils.is_annotated import is_annotated
from .tracking_3d import Tracking3D

if TYPE_CHECKING:
    from ...payload import Payload
    from ...trackers.yolov5_strongsort_osnet_tracker import TrackingResult


class From2DAndDepth(Tracking3D):
    def __call__(self, payload: "Payload") -> "Tuple[Optional[bitarray], Optional[Dict[str, list]]]":
        if not is_annotated(DepthEstimation, payload):
            # payload = payload.filter(DepthEstimation())
            raise Exception()

        if not is_annotated(Tracking2D, payload):
            # payload = payload.filter(Tracking2D())
            raise Exception()

        metadata: "List[Dict[float, Tracking3DResult] | None]" = []
        trajectories: "Dict[float, List[Tracking3DResult]]" = {}

        depths: "List[npt.NDArray | None]" = DepthEstimation.get(payload.metadata)
        trackings: "List[Dict[float, TrackingResult] | None]" = Tracking2D.get(payload.metadata)
        for k, depth, tracking in zip(payload.keep, depths, trackings):
            if not k or tracking is None or depth is None:
                metadata.append(None)
                continue

            trackings3d: "Dict[float, Tracking3DResult]" = {}
            for object_id, t in tracking.items():
                x = int(t.bbox_left + (t.bbox_w / 2))
                y = int(t.bbox_top + (t.bbox_h / 2))
                idx = t.frame_idx
                d = depth[y, x]
                camera = payload.video[idx]
                intrinsic = camera.camera_intrinsic

                point_from_camera = depth_to_3d(x, y, d, intrinsic)
                rotated_offset = Quaternion(camera.camera_rotation).rotate(
                    np.array(point_from_camera)
                )
                point = np.array(camera.camera_translation) + rotated_offset
                trackings3d[object_id] = Tracking3DResult(object_id, point_from_camera, point)
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


@dataclass
class Tracking3DResult:
    object_id: float
    point_from_camera: Tuple[float, float, float]
    point: "npt.NDArray[np.floating]"
    prev: "Tracking3DResult | None" = None
    next: "Tracking3DResult | None" = None
