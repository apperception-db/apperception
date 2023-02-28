#!/usr/bin/env python
# coding: utf-8

# In[1]:




# In[2]:


import json
import os
import pickle
import traceback

import numpy as np

from optimized_ingestion.camera_config import camera_config
from optimized_ingestion.payload import Payload
from optimized_ingestion.pipeline import Pipeline
from optimized_ingestion.stages.decode_frame.parallel_decode_frame import ParallelDecodeFrame
from optimized_ingestion.stages.decode_frame.decode_frame import DecodeFrame
from optimized_ingestion.stages.detection_2d.yolo_detection import YoloDetection
from optimized_ingestion.stages.detection_2d.object_type_filter import ObjectTypeFilter
from optimized_ingestion.stages.detection_3d.from_2d_and_road import From2DAndRoad as FromD2DAndRoad
from optimized_ingestion.stages.filter_car_facing_sideway import FilterCarFacingSideway
from optimized_ingestion.stages.detection_estimation import DetectionEstimation
from optimized_ingestion.stages.detection_estimation.segment_mapping import RoadPolygonInfo
from optimized_ingestion.stages.tracking_2d.strongsort import StrongSORT
from optimized_ingestion.stages.tracking_2d.tracking_2d import Tracking2D, Tracking2DResult
from optimized_ingestion.stages.tracking_3d.from_2d_and_road import From2DAndRoad as FromT2DAndRoad
from optimized_ingestion.stages.tracking_3d.tracking_3d import Tracking3DResult
from optimized_ingestion.stages.segment_trajectory import SegmentTrajectory
from optimized_ingestion.stages.segment_trajectory.construct_segment_trajectory import SegmentPoint
from optimized_ingestion.stages.segment_trajectory.from_tracking_3d import FromTracking3D

from optimized_ingestion.video import Video


# In[3]:


# from optimized_ingestion.cache import disable_cache
# disable_cache()


# In[4]:


BOSTON_VIDEOS = [
#     "scene-0757-CAM_FRONT",
    # "scene-0103-CAM_FRONT",
    # "scene-0553-CAM_FRONT",
    # "scene-0665-CAM_FRONT",
#     "scene-0655-CAM_FRONT_RIGHT",
    "scene-0655-CAM_BACK_RIGHT",
]

NUSCENES_PROCESSED_DATA = "NUSCENES_PROCESSED_DATA"


# In[5]:


import torch


# In[6]:


class DataclassJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, Tracking3DResult):
            return {
                "frame_idx": o.frame_idx,
                "detection_id": tuple(o.detection_id),
                "object_id": o.object_id,
                "point_from_camera": o.point_from_camera,
                "point": o.point.tolist(),
                "bbox_left": o.bbox_left,
                "bbox_top": o.bbox_top,
                "bbox_w": o.bbox_w,
                "bbox_h": o.bbox_h,
                "object_type": o.object_type,
                "timestamp": str(o.timestamp),
            }
        if isinstance(o, Tracking2DResult):
            return {
                "detection_id": tuple(o.detection_id),
                "frame_idx": o.frame_idx,
                "object_id": o.object_id,
                "bbox_left": o.bbox_left,
                "bbox_top": o.bbox_top,
                "bbox_w": o.bbox_w,
                "bbox_h": o.bbox_h,
                "object_type": o.object_type,
                "confidence": o.confidence
            }
        if isinstance(o, SegmentPoint):
            return {
                "detection_id": tuple(o.detection_id),
                "car_loc3d": o.car_loc3d,
                "timestamp": str(o.timestamp),
                "segment_line": str(o.segment_line),
                "segment_heading": o.segment_heading,
                "road_polygon_info": o.road_polygon_info,
                "obj_id": o.obj_id,
                "next": None if o.next is None else tuple(o.next.detection_id),
                "prev": None if o.prev is None else tuple(o.prev.detection_id),
                
            }
        if isinstance(o, RoadPolygonInfo):
            return {
                "id": o.id,
                "polygon": str(o.polygon),
                "segment_lines": [str(l) for l in o.segment_lines],
                "road_type": o.road_type,
                "segment_headings": o.segment_headings,
                "contains_ego": o.contains_ego,
                "ego_config": o.ego_config,
                "fov_lines": o.fov_lines
            }
        if isinstance(o, torch.Tensor):
            return o.tolist()
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


# In[7]:


print(NUSCENES_PROCESSED_DATA in os.environ)
print(os.environ['NUSCENES_PROCESSED_DATA'])


# In[8]:


if NUSCENES_PROCESSED_DATA in os.environ:
    DATA_DIR = os.environ[NUSCENES_PROCESSED_DATA]
else:
    DATA_DIR = "/work/apperception/data/nuScenes/full-dataset-v1.0/Mini"
with open(os.path.join(DATA_DIR, "videos", "frames.pkl"), "rb") as f:
    videos = pickle.load(f)


# In[9]:


def run_benchmark(pipeline, filename, ignore_error=False):
    metadata_strongsort = {}
    metadata_3d = {}
    metadata_segment = {}
    failed_videos = []
    for i, (name, video) in enumerate(videos.items()):
        try:
            print(name, f'---{i} / {len(videos)}---------------------------------------------------------')
            video_filename = video['filename']
            print(video_filename)
            if not video_filename.startswith('boston') or 'FRONT' not in name:
                continue
            frames = Video(
                os.path.join(DATA_DIR, "videos", video["filename"]),
                [camera_config(*f, 0) for f in video["frames"]],
            )

            output = pipeline.run(Payload(frames))
            metadata_strongsort[name] = StrongSORT.get(output)
            metadata_3d[name] = FromT2DAndRoad.get(output)
            metadata_segment[name] = SegmentTrajectory.get(output)
        except Exception as e:
            if ignore_error:
                failed_videos.append((name, str(traceback.format_exc())))
                print('ERRORRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR')
                print('ERRORRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR')
                print('ERRORRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR')
            else:
                raise e

        # Save progress every video
        with open(f"./outputs/{filename}.json", "w") as f:
            json.dump(metadata_strongsort, f, cls=DataclassJSONEncoder, indent=2)

        with open(f"./outputs/{filename}-3d.json", "w") as f:
            json.dump(metadata_3d, f, cls=DataclassJSONEncoder, indent=2)

        with open(f"./outputs/{filename}-segment.json", "w") as f:
            json.dump(metadata_segment, f, cls=DataclassJSONEncoder, indent=2)

        with open(f"./outputs/{filename}-failed-videos.json", "w") as f:
            json.dump(failed_videos, f, indent=2)

        with open(f"./outputs/{filename}-performance.json", "w") as f:
            performance = [
                {
                    "stage": stage.classname(),
                    "runtimes": stage.runtimes,
                }
                for stage
                in pipeline.stages
            ]
            json.dump(performance, f, indent=2)


# In[10]:


type_filter = ['motorcycle', 'car', 'truck', 'bus']


# In[11]:


pipeline1 = Pipeline()
pipeline1.add_filter(ParallelDecodeFrame())
# pipeline1.add_filter(DecodeFrame())
pipeline1.add_filter(filter=YoloDetection())
pipeline1.add_filter(filter=ObjectTypeFilter(type_filter))

pipeline1.add_filter(filter=FromD2DAndRoad())
pipeline1.add_filter(filter=StrongSORT(cache=True))
# pipeline1.add_filter(filter=StrongSORT(cache=False))
pipeline1.add_filter(filter=FromT2DAndRoad())

pipeline1.add_filter(filter=FromTracking3D())

# run_benchmark(pipeline1, 'segment-tracking-without-de', ignore_error=True)


# In[12]:


from optimized_ingestion.stages.depth_estimation import DepthEstimation
from optimized_ingestion.stages.tracking_3d.from_2d_and_depth import From2DAndDepth

pipeline3 = Pipeline()
pipeline3.add_filter(ParallelDecodeFrame())
# pipeline3.add_filter(DecodeFrame())
pipeline3.add_filter(filter=YoloDetection())
pipeline3.add_filter(filter=ObjectTypeFilter(type_filter))

pipeline3.add_filter(filter=FromD2DAndRoad())
pipeline3.add_filter(filter=StrongSORT(cache=True))
pipeline3.add_filter(filter=DepthEstimation())
pipeline3.add_filter(filter=From2DAndDepth())


run_benchmark(pipeline3, 'tracking-with-depth-estimation', ignore_error=True)


# In[ ]:


pipeline2 = Pipeline()
pipeline2.add_filter(ParallelDecodeFrame())
# pipeline2.add_filter(DecodeFrame())
pipeline2.add_filter(filter=YoloDetection())
pipeline2.add_filter(filter=ObjectTypeFilter(type_filter))

pipeline2.add_filter(filter=FromD2DAndRoad())
pipeline2.add_filter(filter=DetectionEstimation())
pipeline2.add_filter(filter=StrongSORT())
pipeline2.add_filter(filter=FromT2DAndRoad())

pipeline2.add_filter(filter=SegmentTrajectory())

# run_benchmark(pipeline2, 'segment-tracking-with-de')


# In[ ]:


with open(f"./outputs/segment-tracking-without-de-performance.json", "r") as f:
    benchmark = json.load(f)

ss_cache, ss_no_cache, *_ = benchmark[4:]
benchmark_data = [
    {
        'name': ssc['name'],
        'runtime_cache': ssc['runtime'],
        'runtime_no_cache': ssnc['runtime']
    }
    for ssc, ssnc
    in zip(ss_cache['runtimes'], ss_no_cache['runtimes'])
]
benchmark_data


# In[ ]:


import altair as alt
import pandas as pd


# In[ ]:


(alt.Chart(pd.DataFrame.from_dict(benchmark_data))
    .mark_point()
    .encode(
        x='runtime_cache:Q',
        y='runtime_no_cache:Q',
        tooltip=['name']
    )
)


# In[ ]:





# In[ ]:




