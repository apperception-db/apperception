import numpy as np
import torch

from ...payload import Payload
from ...utils.depth_to_3d import depth_to_3d
from ..depth_estimation import DepthEstimation
from ..detection_2d.detection_2d import Detection2D
from . import Detection3D, Metadatum


class FromDetection2DAndDepth(Detection3D):
    def _run(self, payload: "Payload"):
        depths = DepthEstimation.get(payload.metadata)
        assert depths is not None

        d2ds = Detection2D.get(payload)
        assert d2ds is not None

        metadata: "list[Metadatum]" = []
        for k, depth, (detections, classes, dids), frame in zip(payload.keep, depths, d2ds, payload.video):
            if not k or len(dids) == 0 or depth is None:
                metadata.append(Metadatum(torch.tensor([], device=detections.device), classes, []))
                continue

            d3ds = []
            for detection in detections:
                bbox_left, bbox_top, bbox_w, bbox_h = detection[:4]

                xc = int(bbox_left + (bbox_w / 2))
                yc = int(bbox_top + (bbox_h / 2))

                xl = int(bbox_left)
                xr = int(bbox_left + bbox_w)

                height, width = depth.shape

                d = depth[
                    max(0, min(yc, height - 1)),
                    max(0, min(xc, width - 1)),
                ]
                intrinsic = frame.camera_intrinsic

                point_from_camera_l = depth_to_3d(xl, yc, d, intrinsic)
                rotated_offset_l = frame.camera_rotation.rotate(
                    np.array(point_from_camera_l)
                )
                point_l = np.array(frame.camera_translation) + rotated_offset_l

                point_from_camera_r = depth_to_3d(xr, yc, d, intrinsic)
                rotated_offset_r = frame.camera_rotation.rotate(
                    np.array(point_from_camera_r)
                )
                point_r = np.array(frame.camera_translation) + rotated_offset_r

                d3d = [*detection, *point_l, *point_r, *point_from_camera_l, *point_from_camera_r]
                d3ds.append(d3d)
            metadata.append(Metadatum(torch.tensor(d3ds, device=detections.device), classes, dids))

        return None, {self.classname(): metadata}
