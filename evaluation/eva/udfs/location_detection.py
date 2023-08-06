from evadb.udfs.abstract.abstract_udf import AbstractUDF
from evadb.udfs.decorators.decorators import forward, setup

import numpy as np
import pandas as pd
import torch
from pyquaternion import Quaternion
from spatialyze.video_processor.stages.decode_frame.decode_frame import DecodeFrame
from spatialyze.video_processor.stages.stage import Stage
from evadb.udfs.abstract.abstract_udf import AbstractUDF
from evadb.udfs.decorators.decorators import forward, setup
from evadb.udfs.decorators.io_descriptors.data_types import PandasDataframe
from evadb.catalog.catalog_type import NdArrayType
from evadb.udfs.gpu_compatible import GPUCompatible

class LocationDetection(AbstractUDF):
    @setup(cacheable=True, udf_type="FeatureExtraction", batchable=True)
    def setup(self):
        pass
    
    @forward(
        input_signatures=[
            PandasDataframe(
                columns=["detections", "depths", "cameraTranslations", "cameraRotations", "cameraIntrinsics"],
                column_types=[NdArrayType.FLOAT32, NdArrayType.FLOAT32, NdArrayType.FLOAT32, NdArrayType.FLOAT32, NdArrayType.FLOAT32],
                column_shapes=[(None,), (None,), (None,), (None,), (None,)],
            )
        ],
        output_signatures=[
            PandasDataframe(
                columns=["locations"],
                column_types=[
                    NdArrayType.FLOAT32,
                ],
                column_shapes=[(None)],
            )
        ],
    )
    def forward(self, df): 
        def _forward(row):
            classes, detections, confs, depth, cameraTranslation, _, cameraIntrinsic = [np.array(x) for x in row.iloc]  
            ## Eva has a bug where duplicate rows sometimes appear with NaN values so we discard these
            if np.isnan(cameraTranslation).any():
                d3ds = [[-1, -1, -1, "dummy", 0]]
                return d3ds
                
                
            cameraRotation = row.iloc[5]

            # Try block not necassary, was just debugging some stuff
            try:
                cameraRotation = np.fromstring(cameraRotation[1:-1], sep=', ')
            except Exception:
                print(cameraTranslation, type(cameraTranslation))
                print(row)
            depth = depth[0]
            d3ds = []
            for (detection, objClass, conf) in zip(detections, classes, confs):
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
                cameraRotationQuat = Quaternion(cameraRotation)
                point_from_camera_l = depth_to_3d(xl, yc, d, cameraIntrinsic)
                rotated_offset_l = cameraRotationQuat.rotate(
                    np.array(point_from_camera_l)
                )
                point_l = np.array(cameraTranslation) + rotated_offset_l

                point_from_camera_r = depth_to_3d(xr, yc, d, cameraIntrinsic)
                rotated_offset_r = cameraRotationQuat.rotate(
                    np.array(point_from_camera_r)
                )
                point_r = np.array(cameraTranslation) + rotated_offset_r

                point_from_camera_c = depth_to_3d(xc, yc, d, cameraIntrinsic)
                rotated_offset_c = cameraRotationQuat.rotate(
                    np.array(point_from_camera_c)
                )
                point_c = np.array(cameraTranslation) + rotated_offset_c
                # if objClass == "car" or objClass == "truck":
                #     print("a", rotated_offset_c)
                #     print(point_c)
                #     print(point_l)
                #     print(cameraTranslation)
                # d3d = [*detection, *point_l, *point_r, *point_from_camera_l, *point_from_camera_r]
                # print(d3d)
                d3d = [*point_c, objClass, conf]
                d3ds.append(d3d)
            return d3ds

        ret = pd.DataFrame()
        ret["locations"] = df.apply(_forward, axis=1)
        return ret

    def name(self):
        return "LocationDetection"
    

from math import sqrt

import numpy as np
import numpy.typing as npt



def depth_to_3d(
    x: float,
    y: float,
    depth: float,
    intrinsic: "npt.NDArray[np.float32]",
):
    [[fx, _, x0], [_, fy, y0], [_, _, s]] = intrinsic

    unit_x: float = (s * x - x0) / fx
    unit_y: float = (s * y - y0) / fy

    Z = depth / sqrt(1 + (unit_x**2) + (unit_y**2))
    X = unit_x * Z
    Y = unit_y * Z
    return X, Y, Z