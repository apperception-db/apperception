from evadb.udfs.abstract.abstract_udf import AbstractUDF
from evadb.udfs.decorators.decorators import forward, setup


class LocationDetection(AbstractUDF):
    @setup(cacheable=True, udf_type="object_detection", batchable=True)
    def setup(self):
        pass
    
    @forward(
        input_signatures=[],
        output_signatures=[],
    )
    def forward(self, detections, depths, camera_config):
        assert depths is not None

        d2ds = detections
        assert d2ds is not None

        metadata: "list[Metadatum]" = []
        for depth, (detections, classes, dids), frame in zip(depths, d2ds, payload.video):
            if not k or len(dids) == 0 or depth is None:
                metadata.append(Metadatum(torch.tensor([], device=detections.device), classes, []))
                continue

            d3ds = []
            for detection in detections:
                bbox_left, bbox_top, bbox_right, bbox_bottom = detection[:4]

                xc = int((bbox_left + bbox_right) / 2)
                yc = int((bbox_top + bbox_bottom) / 2)

                xl = int(bbox_left)
                xr = int(bbox_right)

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

    def name(self):
        return "LocationDetection"
    

from math import sqrt

import numpy as np
import numpy.typing as npt

from ..types import Float3, Float33


def depth_to_3d(
    x: float,
    y: float,
    depth: float,
    intrinsic: "npt.NDArray[np.float32] | Float33",
) -> "Float3":
    [[fx, _, x0], [_, fy, y0], [_, _, s]] = intrinsic

    unit_x: float = (s * x - x0) / fx
    unit_y: float = (s * y - y0) / fy

    Z = depth / sqrt(1 + (unit_x**2) + (unit_y**2))
    X = unit_x * Z
    Y = unit_y * Z
    return X, Y, Z