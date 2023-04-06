import numpy as np
from bitarray import bitarray
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from ...utils.depth_to_3d import depth_to_3d
from ..depth_estimation import DepthEstimation
from ..tracking_2d.tracking_2d import Tracking2D
from .tracking_3d import Tracking3D, Tracking3DResult

if TYPE_CHECKING:
    from ...payload import Payload

    # from ...trackers.yolov5_strongsort_osnet_tracker import TrackingResult


class From2DAndDepth(Tracking3D):
    def _run(self, payload: "Payload") -> "Tuple[Optional[bitarray], Optional[Dict[str, list]]]":
        metadata: "List[Dict[int, Tracking3DResult] | None]" = []
        trajectories: "Dict[int, List[Tracking3DResult]]" = {}

        depths = DepthEstimation.get(payload.metadata)
        assert depths is not None

        trackings = Tracking2D.get(payload.metadata)
        assert trackings is not None

        for k, depth, tracking, frame in zip(payload.keep, depths, trackings, payload.video):
            if not k or tracking is None or depth is None:
                metadata.append(None)
                continue

            trackings3d: "Dict[int, Tracking3DResult]" = {}
            for object_id, t in tracking.items():
                x = int(t.bbox_left + (t.bbox_w / 2))
                y = int(t.bbox_top + (t.bbox_h / 2))
                idx = t.frame_idx
                height, width = depth.shape
                d = depth[min(y, height - 1), min(x, width - 1)]
                camera = payload.video[idx]
                intrinsic = camera.camera_intrinsic

                point_from_camera = depth_to_3d(x, y, d, intrinsic)
                rotated_offset = camera.camera_rotation.rotate(
                    np.array(point_from_camera)
                )
                point = np.array(camera.camera_translation) + rotated_offset
                trackings3d[object_id] = Tracking3DResult(
                    t.frame_idx,
                    t.detection_id,
                    t.object_id,
                    point_from_camera,
                    point,
                    t.bbox_left,
                    t.bbox_top,
                    t.bbox_w,
                    t.bbox_h,
                    t.object_type,
                    frame.timestamp
                )
                if object_id not in trajectories:
                    trajectories[object_id] = []
                trajectories[object_id].append(trackings3d[object_id])
            metadata.append(trackings3d)

        for trajectory in trajectories.values():
            last = len(trajectory) - 1
            for i, traj in enumerate(trajectory):
                if i > 0:
                    traj.prev = trajectory[i - 1]
                if i < last:
                    traj.next = trajectory[i + 1]

        return None, {self.classname(): metadata}
