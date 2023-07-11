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

class IntersectionQuery(AbstractUDF):
    @setup(cacheable=True, udf_type="Query", batchable=True)
    def setup(self):
        self.nusc_map = NuScenesMap(dataroot='/data/raw/map-expansion', map_name='boston-seaport')
    
    @forward(
        input_signatures=[
            PandasDataframe(
                columns=["locations"],
                column_types=[NdArrayType.ANYTYPE],
                column_shapes=[(None,)],
            )
        ],
        output_signatures=[
            PandasDataframe(
                columns=["queryresult"],
                column_types=[NdArrayType.ANYTYPE],
                column_shapes=[(None)],
            )
        ],
    )
    def forward(self, df):
        objClasses = ["truck"]
        def _forward(row):
            locations, egoTranslation, egoHeading = [np.array(x) for x in row.iloc]
            for object in locations:
                x, y, z, objClass, conf = object
                loc = [float(x) for x in (x, y, z)]

                if objClass not in objClasses:
                   continue
                
                convX, convY = convert_camera([np.float(x) for x in egoTranslation], np.float(egoHeading), loc)

                if -10 < convX < -1 and 0 < convY < 50:
                   return True
            return False

        ret = pd.DataFrame()
        ret["queryresult"] = df.apply(_forward, axis=1)
        return ret


    def name(self):
        return "RelativePositionQuery"

def convert_camera(cam_position, cam_heading, obj_point):
    cam_x, cam_y, _ = cam_position
    obj_x, obj_y, _ = obj_point
    
    subtract_x = obj_x - cam_x
    subtract_y = obj_y - cam_y

    subtract_mag = np.sqrt(subtract_x**2 + subtract_y**2)

    res_x = subtract_mag * np.cos(-cam_heading + np.arctan2(subtract_y, subtract_x))
    res_y = subtract_mag * np.sin(-cam_heading + np.arctan2(subtract_y, subtract_x))

    return res_x, res_y