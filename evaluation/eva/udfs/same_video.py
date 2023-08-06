from evadb.udfs.abstract.abstract_udf import AbstractUDF
from evadb.udfs.decorators.decorators import forward, setup
import numpy as np
import pandas as pd
import torch
from pyquaternion import Quaternion
from optimized_ingestion.stages.decode_frame.decode_frame import DecodeFrame
from optimized_ingestion.stages.stage import Stage
from evadb.udfs.abstract.abstract_udf import AbstractUDF
from evadb.udfs.decorators.decorators import forward, setup
from evadb.udfs.decorators.io_descriptors.data_types import PandasDataframe
from evadb.catalog.catalog_type import NdArrayType, ColumnType
from evadb.udfs.gpu_compatible import GPUCompatible
from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.map_expansion import arcline_path_utils
from nuscenes.map_expansion.bitmap import BitMap

class SameVideo(AbstractUDF):
    @setup(cacheable=True, udf_type="object_detection", batchable=True)
    def setup(self):
       pass 

    @forward(
        input_signatures=[
            PandasDataframe(
                columns=["videofile", "cameraid"],
                column_types=[NdArrayType.ANYTYPE, NdArrayType.ANYTYPE],
                column_shapes=[(None,), (None,)],
            )
        ],
        output_signatures=[
            PandasDataframe(
                columns=["issame"],
                column_types=[NdArrayType.ANYTYPE],
                column_shapes=[(None)],
            )
        ],
    )
    def forward(self, df):
        def _forward(row):
            videofile, cameraid = row.iloc
            return cameraid in videofile

        ret = pd.DataFrame()
        ret["issame"] = df.apply(_forward, axis=1)
        return ret


    def name(self):
        return "SameVideo"
