#!/usr/bin/env python
# coding: utf-8

# In[1]:


import subprocess
import json
import os
import pickle
import traceback
import shutil
import socket
import time
import random

from os import environ

import numpy as np
import torch
import psycopg2

subprocess.Popen('nvidia-smi', shell=True).wait()
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
    from nbutils.report_progress import report_progress
else:
    from tqdm import tqdm
    from playground.nbutils.report_progress import report_progress


# In[4]:


process.wait()


# In[5]:


from optimized_ingestion.camera_config import camera_config
from optimized_ingestion.payload import Payload
from optimized_ingestion.pipeline import Pipeline
from optimized_ingestion.video import Video
from optimized_ingestion.metadata_json_encoder import MetadataJSONEncoder


# In[6]:


# Stages
from optimized_ingestion.stages.in_view import InView

from optimized_ingestion.stages.decode_frame.decode_frame import DecodeFrame

from optimized_ingestion.stages.detection_2d.detection_2d import Detection2D
from optimized_ingestion.stages.detection_2d.yolo_detection import YoloDetection
from optimized_ingestion.stages.detection_2d.object_type_filter import ObjectTypeFilter
from optimized_ingestion.stages.detection_2d.ground_truth import GroundTruthDetection


# In[7]:


from optimized_ingestion.stages.detection_3d.from_detection_2d_and_road import FromDetection2DAndRoad
from optimized_ingestion.stages.detection_3d.from_detection_2d_and_depth import FromDetection2DAndDepth

from optimized_ingestion.stages.depth_estimation import DepthEstimation

from optimized_ingestion.stages.detection_estimation import DetectionEstimation
from optimized_ingestion.stages.detection_estimation.segment_mapping import RoadPolygonInfo


# In[8]:


from optimized_ingestion.stages.tracking.strongsort import StrongSORT
from optimized_ingestion.stages.tracking_2d.strongsort import StrongSORT as StrongSORT2D


# In[9]:


from optimized_ingestion.stages.tracking_3d.from_tracking_2d_and_road import FromTracking2DAndRoad
from optimized_ingestion.stages.tracking_3d.from_tracking_2d_and_depth import FromTracking2DAndDepth
from optimized_ingestion.stages.tracking_3d.tracking_3d import Tracking3DResult, Tracking3D

from optimized_ingestion.stages.segment_trajectory import SegmentTrajectory
from optimized_ingestion.stages.segment_trajectory.construct_segment_trajectory import SegmentPoint
from optimized_ingestion.stages.segment_trajectory.from_tracking_3d import FromTracking3D


# In[10]:


from optimized_ingestion.cache import disable_cache
disable_cache()


# In[11]:


from optimized_ingestion.utils.process_pipeline import format_trajectory, insert_trajectory, get_tracks
from optimized_ingestion.actions.tracking2d_overlay import tracking2d_overlay


# In[12]:


from apperception.utils.ingest_road import ingest_road
from apperception.database import database, Database
from apperception.world import empty_world
from apperception.utils import F
from apperception.predicate import camera, objects, lit, FindAllTablesVisitor, normalize, MapTablesTransformer, GenSqlVisitor
from apperception.data_types.camera import Camera as ACamera
from apperception.data_types.camera_config import CameraConfig as ACameraConfig


# In[13]:


NUSCENES_PROCESSED_DATA = "NUSCENES_PROCESSED_DATA"
print(NUSCENES_PROCESSED_DATA in os.environ)
print(os.environ['NUSCENES_PROCESSED_DATA'])


# In[14]:


DATA_DIR = os.environ[NUSCENES_PROCESSED_DATA]
# with open(os.path.join(DATA_DIR, "videos", "frames.pkl"), "rb") as f:
#     videos = pickle.load(f)
with open(os.path.join(DATA_DIR, 'videos', 'videos.json'), 'r') as f:
    videos = json.load(f)


# In[15]:


with open('./data/evaluation/video-samples/boston-seaport.txt', 'r') as f:
    sampled_scenes = f.read().split('\n')
print(sampled_scenes[0], sampled_scenes[-1], len(sampled_scenes))


# In[16]:


BENCHMARK_DIR = "./outputs/run"


def bm_dir(*args: "str"):
    return os.path.join(BENCHMARK_DIR, *args)


# In[17]:


def get_sql(predicate: "PredicateNode"):
    tables, camera = FindAllTablesVisitor()(predicate)
    tables = sorted(tables)
    mapping = {t: i for i, t in enumerate(tables)}
    predicate = normalize(predicate)
    predicate = MapTablesTransformer(mapping)(predicate)

    t_tables = ''
    t_outputs = ''
    for i in range(len(tables)):
        t_tables += '\n' \
            'JOIN Item_General_Trajectory ' \
            f'AS t{i} ' \
            f'ON Cameras.timestamp <@ t{i}.trajCentroids::period'
        t_outputs += f', t{i}.itemId'

    return f"""
        SELECT Cameras.frameNum {t_outputs}
        FROM Cameras{t_tables}
        WHERE
        {GenSqlVisitor()(predicate)}
    """


# In[18]:


slices = {
    "noopt": (0, 1),
    "inview": (1, 2),
    "objectfilter": (2, 3),
    "geo": (3, 4),
    "de": (4, 5),
    "opt": (5, 6),
    # "optde": (6, 7),
    'dev': (0, 2),
    'freddie': (1, 2),
}


# In[19]:


def run_benchmark(pipeline, filename, run=0, ignore_error=False):
    print(filename)
    metadata_strongsort = {}
    metadata_d2d = {}
    failed_videos = []

    all_metadata = {
        'detection': metadata_d2d,
        'sort': metadata_strongsort,
    }
    print('# of total    videos:', len(videos))

    names = set(sampled_scenes[:50])
    # names = {'0655'}
    filtered_videos = [
        n for n in videos
        if n[6:10] in names and 'FRONT' in n
    ]
    N = len(filtered_videos)
    print('# of filtered videos:', N)

    s_from, s_to = slices[test]
    STEP = N // 6
    print('test', test)
    print('from', s_from*STEP)
    print('to  ', s_to*STEP)
    filtered_videos = filtered_videos[s_from*STEP:s_to*STEP]
    print('# of sliced   videos:', len(filtered_videos))
    # ingest_road(database, './data/scenic/road-network/boston-seaport')

    for pre in [*all_metadata.keys(), 'qresult', 'performance', 'failedvideos']:
        p = os.path.join(BENCHMARK_DIR, f"{pre}--{filename}_{run}")
        if os.path.exists(p):
            shutil.rmtree(p)
        os.makedirs(p)

    def save_perf():
        for n, message in failed_videos:
            p = bm_dir(f'failedvideos--{filename}_{run}', f'{n}.txt')
            with open(p, "w") as f:
                f.write(message)

    for i, name in tqdm(enumerate(filtered_videos), total=len(filtered_videos)):
        try:
            start_input = time.time()
            with open(os.path.join(DATA_DIR, 'videos', 'boston-seaport-' + name + '.pkl'), 'rb') as f:
                video = pickle.load(f)
            video_filename = video['filename']

            frames = Video(
                os.path.join(DATA_DIR, "videos", video["filename"]),
                [camera_config(*f, 0) for f in video["frames"]],
            )
            time_input = time.time() - start_input

            output = pipeline.run(Payload(frames))

            metadata_strongsort[name] = output[StrongSORT2D]
            metadata_d2d[name] = output[Detection2D]

            for pre, metadata in all_metadata.items():
                p = bm_dir(f"{pre}--{filename}_{run}", f"{name}.json")
                with open(p, "w") as f:
                    json.dump(metadata[name], f, cls=MetadataJSONEncoder, indent=1)

            perf = []
            for stage in pipeline.stages:
                benchmarks = [*filter(
                    lambda x: video['filename'] in x['name'],
                    stage.benchmark
                )]
                assert len(benchmarks) == 1
                perf.append({
                    'stage': stage.classname(),
                    'benchmark': benchmarks[0]
                })
            p = bm_dir(f'performance--{filename}_{run}', f'{name}.json')
            with open(p, "w") as f:
                json.dump(perf, f, indent=1)
        except Exception as e:
            if ignore_error:
                message = str(traceback.format_exc())
                failed_videos.append((name, message))
                print(video_filename)
                print(e)
                print(message)
                print("------------------------------------------------------------------------------------")
                print()
                print()
            else:
                raise e

        if len(metadata_d2d) % 10 == 0:
            save_perf()
    save_perf()


# In[20]:


def create_pipeline(
    ss_cache,
):
    pipeline = Pipeline()

    # Decode
    pipeline.add_filter(DecodeFrame())

    # 2D Detection
    pipeline.add_filter(YoloDetection())

    # Tracking
    pipeline.add_filter(StrongSORT2D(
        # method='update-empty' if ss_update_when_skip else 'increment-ages',
        method='update-empty',
        cache=ss_cache
    ))

    return pipeline


# In[21]:


# if test == 'dev':
#     test = 'opt'


# In[22]:


def run(__test):
    p2 = create_pipeline(ss_cache=(_test == 'opt'))

    print(p2)
    run_benchmark(p2, 'sscache' + __test, run=1, ignore_error=True)


# In[23]:


# tests = ['noopt', 'inview', 'objectfilter', 'geo', 'de', 'opt', 'optde']
tests = ['opt', 'noopt']
random.shuffle(tests)

# for _test in tests:
#     assert isinstance(pipelines[_test](lit(True)), Pipeline)

for idx, _test in enumerate(tests):
    print(f'----------- {idx} / {len(tests)} --- {_test} -----------')
    run(_test)


# In[ ]:


# run(test)


# In[ ]:


# if test == 'opt':
#     run('optde')


# In[ ]:


if not is_notebook():
    subprocess.Popen('sudo shutdown -h now', shell=True)


# In[ ]:




