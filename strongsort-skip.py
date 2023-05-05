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
from optimized_ingestion.metadata_json_encoder import MetadataJSONEncoder


# In[6]:


# Stages
from optimized_ingestion.stages.decode_frame.decode_frame import DecodeFrame

from optimized_ingestion.stages.detection_2d.detection_2d import Detection2D
from optimized_ingestion.stages.detection_2d.yolo_detection import YoloDetection


# In[7]:


from optimized_ingestion.stages.strongsort_with_skip import StrongSORTWithSkip


# In[8]:


from optimized_ingestion.cache import disable_cache
disable_cache()


# In[9]:


NUSCENES_PROCESSED_DATA = "NUSCENES_PROCESSED_DATA"
print(NUSCENES_PROCESSED_DATA in os.environ)
print(os.environ['NUSCENES_PROCESSED_DATA'])


# In[10]:


DATA_DIR = os.environ[NUSCENES_PROCESSED_DATA]
with open(os.path.join(DATA_DIR, "videos", "frames.pkl"), "rb") as f:
    videos = pickle.load(f)


# In[11]:


with open(os.path.join(DATA_DIR, 'cities.pkl'), 'rb') as f:
    cities = pickle.load(f)


# In[12]:


BENCHMARK_DIR = "./outputs/ablation-performance-run"


def bm_dir(*args: "str"):
    return os.path.join(BENCHMARK_DIR, *args)


# In[13]:


def run_benchmark(pipeline, filename, run=0, ignore_error=False):
    metadata_strongsort = {}
    failed_videos = []

    all_metadata = {
        'sort': metadata_strongsort,
    }
    names = cities['boston-seaport'][int(test) * 20:(int(test) + 1) * 20]
    filtered_videos = [(n, v) for n, v in videos.items() if n[6:10] in names]

    for pre in all_metadata.keys():
        p = os.path.join(BENCHMARK_DIR, f"{pre}--{filename}_{run}")
        if os.path.exists(p):
            shutil.rmtree(p)
        os.makedirs(p)

    for i, (name, video) in tqdm(enumerate(filtered_videos), total=len(filtered_videos)):
        try:
            video_filename = video['filename']
            if not video_filename.startswith('boston') or 'FRONT' not in name:
                continue

            frames = Video(
                os.path.join(DATA_DIR, "videos", video["filename"]),
                [camera_config(*f, 0) for f in video["frames"]],
            )

            output = pipeline.run(Payload(frames))

            metadata_strongsort[name] = output[StrongSORTWithSkip]

            for pre, metadata in all_metadata.items():
                p = bm_dir(f"{pre}--{filename}_{run}", f"{name}.json")
                with open(p, "w") as f:
                    json.dump(metadata[name], f, cls=MetadataJSONEncoder, indent=1)
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

        if i % 20 == 0:
            with open(bm_dir(f"failed_videos--{filename}_{run}.json"), "w") as f:
                json.dump(failed_videos, f, indent=1)

            with open(bm_dir(f"perf--{filename}_{run}.json"), "w") as f:
                performance = [
                    {
                        "stage": stage.classname(),
                        "benchmark": stage.benchmark,
                        **(
                            {'explains': stage.explains}
                            if hasattr(stage, 'explains')
                            else {}
                        )
                    }
                    for stage
                    in pipeline.stages
                ]
                json.dump(performance, f, indent=1)


# In[14]:


pipeline = Pipeline()

# Decode
pipeline.add_filter(DecodeFrame())
# 2D Detection
pipeline.add_filter(YoloDetection())
# Tracking
pipeline.add_filter(StrongSORTWithSkip())

for i in range(3):
    run_benchmark(pipeline, 'ss-skip-benchmark', run=i, ignore_error=True)


# In[ ]:


if not is_notebook():
    subprocess.Popen('sudo shutdown -h now', shell=True)

