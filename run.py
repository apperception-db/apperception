#!/usr/bin/env python
# coding: utf-8

# In[1]:


import subprocess
import json
import os
import pickle
import traceback
import socket

import numpy as np
import torch


process = subprocess.Popen('docker container start mobilitydb', shell=True)


# In[2]:


hostname = socket.gethostname()
test = hostname.split("-")[-1]
print("test", test)


# In[3]:


def is_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            # Jupyter notebook or qtconsole
            return True
        elif shell == 'TerminalInteractiveShell':
            # Terminal running IPython
            return False
        else:
            # Other type (?)
            return False
    except NameError:
        # Probably standard Python interpreter
        return False


if is_notebook():
    get_ipython().run_line_magic('cd', '..')
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


# In[4]:


process.wait()


# In[5]:


from optimized_ingestion.camera_config import camera_config
from optimized_ingestion.payload import Payload
from optimized_ingestion.pipeline import Pipeline
from optimized_ingestion.video import Video


# In[6]:


# Stages
from optimized_ingestion.stages.in_view import InView

from optimized_ingestion.stages.decode_frame.decode_frame import DecodeFrame

from optimized_ingestion.stages.detection_2d.yolo_detection import YoloDetection
from optimized_ingestion.stages.detection_2d.object_type_filter import ObjectTypeFilter
from optimized_ingestion.stages.detection_2d.ground_truth import GroundTruthDetection

from optimized_ingestion.stages.detection_3d.from_detection_2d_and_road import FromDetection2DAndRoad
from optimized_ingestion.stages.detection_3d.from_detection_2d_and_depth import FromDetection2DAndDepth

from optimized_ingestion.stages.depth_estimation import DepthEstimation

from optimized_ingestion.stages.detection_estimation import DetectionEstimation
from optimized_ingestion.stages.detection_estimation.segment_mapping import RoadPolygonInfo

from optimized_ingestion.stages.tracking_2d.strongsort import StrongSORT
from optimized_ingestion.stages.tracking_2d.tracking_2d import Tracking2D, Tracking2DResult

from optimized_ingestion.stages.tracking_3d.from_tracking_2d_and_road import FromTracking2DAndRoad
from optimized_ingestion.stages.tracking_3d.from_tracking_2d_and_depth import FromTracking2DAndDepth
from optimized_ingestion.stages.tracking_3d.tracking_3d import Tracking3DResult, Tracking3D

from optimized_ingestion.stages.segment_trajectory import SegmentTrajectory
from optimized_ingestion.stages.segment_trajectory.construct_segment_trajectory import SegmentPoint
from optimized_ingestion.stages.segment_trajectory.from_tracking_3d import FromTracking3D


# In[7]:


from optimized_ingestion.cache import disable_cache
disable_cache()


# In[8]:


NUSCENES_PROCESSED_DATA = "NUSCENES_PROCESSED_DATA"


# In[9]:


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
                "segment_line": None if o.segment_line is None else o.segment_line.to_ewkb(),
                # "segment_line_wkb": o.segment_line.wkb_hex,
                "segment_heading": o.segment_heading,
                "road_polygon_info": o.road_polygon_info,
                "obj_id": o.obj_id,
                "type": o.type,
                "next": None if o.next is None else tuple(o.next.detection_id),
                "prev": None if o.prev is None else tuple(o.prev.detection_id),
            }
        if isinstance(o, RoadPolygonInfo):
            return {
                "id": o.id,
                "polygon": str(o.polygon),
                # "polygon_wkb": o.polygon.wkb_hex,
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


# In[10]:


print(NUSCENES_PROCESSED_DATA in os.environ)
print(os.environ['NUSCENES_PROCESSED_DATA'])


# In[11]:


DATA_DIR = os.environ[NUSCENES_PROCESSED_DATA]
with open(os.path.join(DATA_DIR, "videos", "frames.pkl"), "rb") as f:
    videos = pickle.load(f)


# In[12]:


with open(os.path.join(DATA_DIR, 'cities.pkl'), 'rb') as f:
    cities = pickle.load(f)


# In[13]:


def run_benchmark(pipeline, filename, ignore_error=False):
    metadata_strongsort = {}
    metadata_3d = {}
    metadata_segment = {}
    metadata_frame_id = {}
    failed_videos = []

    names = cities['boston-seaport'][:200]
    filtered_videos = [(n, v) for n, v in videos.items() if n[6:10] in names]

    for i, (name, video) in tqdm(enumerate(filtered_videos), total=len(filtered_videos)):
        try:
            video_filename = video['filename']
            # print(video_filename)
            if not video_filename.startswith('boston') or 'FRONT' not in name:
                continue

            frames = Video(
                os.path.join(DATA_DIR, "videos", video["filename"]),
                [camera_config(*f, 0) for f in video["frames"]],
            )

            output = pipeline.run(Payload(frames))
            metadata_strongsort[name] = output[StrongSORT]
            metadata_3d[name] = output[Tracking3D]
            metadata_segment[name] = output[SegmentTrajectory]
            # metadata_frame_id[name] = output[GetCameraConfigId]
        except Exception as e:
            if ignore_error:
                message = str(traceback.format_exc())
                failed_videos.append((name, message))
                print(e)
                print(message)
                print("------------------------------------------------------------------------------------")
                print()
                print()
            else:
                raise e

        # Save progress every video
        with open(f"./outputs/sort--{filename}.json", "w") as f:
            json.dump(metadata_strongsort, f, cls=DataclassJSONEncoder, indent=2)

#         with open(f"./outputs/frame-id--{filename}.json", "w") as f:
#             json.dump(metadata_frame_id, f, indent=2)

#         with open(f"./outputs/tracking-3d--{filename}.json", "w") as f:
#             json.dump(metadata_3d, f, cls=DataclassJSONEncoder, indent=2)

        with open(f"./outputs/segment-trajectory--{filename}.json", "w") as f:
            json.dump(metadata_segment, f, cls=DataclassJSONEncoder, indent=2)

        with open(f"./outputs/failed_videos--{filename}.json", "w") as f:
            json.dump(failed_videos, f, indent=2)

        with open(f"./outputs/perf--{filename}.json", "w") as f:
            performance = [
                {
                    "stage": stage.classname(),
                    "benchmark": stage.benchmark,
                }
                for stage
                in pipeline.stages
            ]
            json.dump(performance, f, indent=2)


# In[14]:


def create_pipeline(
    predicate,
    in_view=True,
    object_filter=True,
    groundtruth_detection=False,
    geo_depth=True,
    detection_estimation=True,
    strongsort=False,
):
    pipeline = Pipeline()

    # In-View Filter
    if in_view:
        # TODO: view angle and road type should depends on the predicate
        pipeline.add_filter(InView(50, 'intersection'))

    # Decode
    pipeline.add_filter(DecodeFrame())

    # 2D Detection
    if groundtruth_detection:
        pass
    else:
        pipeline.add_filter(YoloDetection())

    # Object Filter
    if object_filter:
        # TODO: filter objects based on predicate
        pipeline.add_filter(ObjectTypeFilter(['car']))

    # 3D Detection
    if geo_depth:
        pipeline.add_filter(FromDetection2DAndRoad())
    else:
        pipeline.add_filter(DepthEstimation())
        pipeline.add_filter(FromDetection2DAndDepth())

    # Detection Estimation
    if detection_estimation:
        pipeline.add_filter(DetectionEstimation())

    # Tracking
    pipeline.add_filter(StrongSORT(cache=strongsort))

    # Tracking 3D
    if geo_depth:
        pipeline.add_filter(FromTracking2DAndRoad())
    else:
        pipeline.add_filter(FromTracking2DAndDepth())

    # Segment Trajectory
    pipeline.add_filter(FromTracking3D())

    return pipeline


# In[15]:


predicate = None

p_noOpt = lambda predicate: create_pipeline(
    predicate,
    in_view=False,
    object_filter=False,
    geo_depth=False,
    detection_estimation=False
)

p_inview = lambda predicate: create_pipeline(
    predicate,
    in_view=True,
    object_filter=False,
    geo_depth=False,
    detection_estimation=False
)

p_objectFilter = lambda predicate: create_pipeline(
    predicate,
    in_view=False,
    object_filter=True,
    geo_depth=False,
    detection_estimation=False
)

p_geo = lambda predicate: create_pipeline(
    predicate,
    in_view=False,
    object_filter=False,
    geo_depth=True,
    detection_estimation=False
)

p_de = lambda predicate: create_pipeline(
    predicate,
    in_view=False,
    object_filter=False,
    geo_depth=False,
    detection_estimation=True
)

p_opt = lambda predicate: create_pipeline(
    predicate,
    in_view=True,
    object_filter=True,
    geo_depth=True,
    detection_estimation=False
)

p_optDe = lambda predicate: create_pipeline(
    predicate,
    in_view=True,
    object_filter=True,
    geo_depth=True,
    detection_estimation=True
)

pipelines = {
    "noopt": p_noOpt,
    "inview": p_inview,
    "objectfilter": p_objectFilter,
    "geo": p_geo,
    "de": p_de,
    "opt": p_opt,
    "optde": p_optDe,
}


# In[ ]:


for i in range(10):
    run_benchmark(pipelines[test](None), test + '_' + str(i), ignore_error=True)


# In[ ]:


# subprocess.Popen('shutdown -h now', shell=True)


# In[ ]:




