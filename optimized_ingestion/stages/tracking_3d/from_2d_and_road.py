from typing import Dict, List

import cv2
from pyquaternion import Quaternion

from ...payload import Payload
from ...trackers.yolov5_strongsort_osnet_tracker import TrackingResult
from ..stage import StageOutput
from ..tracking_2d import Tracking2D
from ..utils.is_annotated import is_annotated
from .tracking_3d import Tracking3D, Tracking3DResult


class From2DAndRoad(Tracking3D):
    def __call__(self, payload: "Payload") -> "StageOutput":
        if not is_annotated(Tracking2D, payload):
            # payload = payload.filter(Tracking2D())
            raise Exception()

        metadata: "List[Dict[float, Tracking3DResult] | None]" = []
        trajectories: "Dict[float, List[Tracking3DResult]]" = {}
        video = payload.video
        cap = cv2.VideoCapture(video.videofile)
        trackings: "List[Dict[float, TrackingResult] | None]" = Tracking2D.get(payload.metadata)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        for i, (k, tracking, frame) in enumerate(zip(payload.keep, trackings, video)):
            if not k or tracking is None or len(tracking) == 0:
                metadata.append(None)
                continue

            tracking3d: "Dict[float, Tracking3DResult]" = {}
            [[fx, _, x0], [_, fy, y0], [_, _, s]] = frame.camera_intrinsic
            rotation = Quaternion(frame.camera_rotation)

            for object_id, t in tracking.items():
                y = int(t.bbox_top + t.bbox_h)
                xs = [*range(int(t.bbox_left), int(t.bbox_left + t.bbox_w) + 1)]
                idx = t.frame_idx

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
