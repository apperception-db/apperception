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
        objClasses = ["car", "truck"]
        def _forward(row):
            locations = [np.array(x) for x in row.iloc][0]
            for object in locations:
                x, y, z, objClass, conf = object
                loc = [float(x) for x in (x, y, z)]

                if objClass not in objClasses:
                   continue
                   
                if is_contained_intersection(self.nusc_map, loc):
                   return True
            return False

        ret = pd.DataFrame()
        ret["queryresult"] = df.apply(_forward, axis=1)
        return ret


    def name(self):
        return "IntersectionQuery"

def is_contained_intersection(nusc_map, position):
  x, y, z = position
  road_segment_token = nusc_map.layers_on_point(x, y)['road_segment']
  if road_segment_token != '':
    road_segment = nusc_map.get('road_segment', road_segment_token)
    return road_segment["is_intersection"]
  else:
    return False