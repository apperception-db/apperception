from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
from bitarray import bitarray
from pyquaternion import Quaternion

from ..payload import Payload
from ..utils.depth_to_3d import depth_to_3d
from .depth_estimation import DepthEstimation
from .stage import Stage
from .tracking_2d import Tracking2D
from .utils.is_filtered import is_filtered

if TYPE_CHECKING:
    from ..trackers.yolov5_strongsort_osnet_tracker import TrackingResult


class Tracking3DFrom2DAndDepth(Stage):
    def __call__(self, payload: "Payload") -> "Tuple[Optional[bitarray], Optional[list]]":
        if not is_filtered(DepthEstimation, payload):
            payload = payload.filter(DepthEstimation())

        if not is_filtered(Tracking2D, payload):
            payload = payload.filter(Tracking2D())

        metadata = []
        for k, m in zip(payload.keep, payload.metadata):
            if k:
                depth = DepthEstimation.get(m)
                trackings: "List[TrackingResult]" = Tracking2D.get(m)
                trackings3d: "List[Tracking3DResult]" = []
                for t in trackings:
                    x = int(t.bbox_left + (t.bbox_w / 2))
                    y = int(t.bbox_top + (t.bbox_h / 2))
                    idx = t.frame_idx
                    depth = depth[y, x]
                    camera = payload.video[idx]
                    intrinsic = camera.camera_intrinsic

                    point_from_camera = depth_to_3d(x, y, depth, intrinsic)
                    rotated_offset = Quaternion(camera.camera_rotation).rotate(
                        np.array(point_from_camera)
                    )
                    point = np.array(camera.camera_translation) + rotated_offset
                    trackings3d.append(Tracking3DResult(point_from_camera, point))
                metadata.append({self.classname(): trackings3d})
            else:
                metadata.append(None)

        return None, metadata


@dataclass
class Tracking3DResult:
    point_from_camera: Tuple[float, float, float]
    point: "npt.NDArray[np.floating]"
