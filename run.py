#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import subprocess
import json
import os
import pickle
import traceback
import shutil
import socket

import numpy as np
import torch

subprocess.Popen('nvidia-smi', shell=True).wait()
process = subprocess.Popen('docker container start mobilitydb', shell=True)


# In[ ]:


hostname = socket.gethostname()
test = hostname.split("-")[-1]
print("test", test)


# In[ ]:


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


# In[ ]:


process.wait()


# In[ ]:


from optimized_ingestion.camera_config import camera_config
from optimized_ingestion.payload import Payload
from optimized_ingestion.pipeline import Pipeline
from optimized_ingestion.video import Video
from optimized_ingestion.metadata_json_encoder import MetadataJSONEncoder
from optimized_ingestion.stages.stage import Stage


# In[ ]:


# Stages
from optimized_ingestion.stages.decode_frame.decode_frame import DecodeFrame

from optimized_ingestion.stages.detection_2d.detection_2d import Detection2D
from optimized_ingestion.stages.detection_2d.yolo_detection import YoloDetection


# In[ ]:


from optimized_ingestion.stages.strongsort_cache_benchmark import StrongSORTCacheBenchmark


# In[ ]:


from optimized_ingestion.cache import disable_cache
disable_cache()


# In[ ]:


NUSCENES_PROCESSED_DATA = "NUSCENES_PROCESSED_DATA"
print(NUSCENES_PROCESSED_DATA in os.environ)
print(os.environ['NUSCENES_PROCESSED_DATA'])


# In[ ]:


DATA_DIR = os.environ[NUSCENES_PROCESSED_DATA]
with open(os.path.join(DATA_DIR, "videos", "frames.pkl"), "rb") as f:
    videos = pickle.load(f)


# In[ ]:


with open(os.path.join(DATA_DIR, 'cities.pkl'), 'rb') as f:
    cities = pickle.load(f)


# In[ ]:


BENCHMARK_DIR = "./outputs/run"


def bm_dir(*args: "str"):
    return os.path.join(BENCHMARK_DIR, *args)


# In[ ]:


def run_benchmark(pipeline, filename, run=0, ignore_error=False):
    metadata_strongsort = {}
    failed_videos = []

    all_metadata = {
        'sort': metadata_strongsort,
    }

    names = cities['boston-seaport']
    filtered_videos = [(n, v) for n, v in videos.items() if n[6:10] in names]

    for pre in all_metadata.keys():
        p = os.path.join(BENCHMARK_DIR, f"{pre}--{filename}_{run}")
        if os.path.exists(p):
            shutil.rmtree(p)
        os.makedirs(p)

    for i, (name, video) in tqdm(enumerate(filtered_videos), total=len(filtered_videos)):
    # for i, (name, video) in enumerate(filtered_videos):
        if i % int(len(filtered_videos) / 200) == 0:
            report_progress(i, len(filtered_videos), '')
        try:
            video_filename = video['filename']
            if not video_filename.startswith('boston') or 'FRONT' not in name:
                continue

            frames = Video(
                os.path.join(DATA_DIR, "videos", video["filename"]),
                [camera_config(*f, 0) for f in video["frames"]],
            )

            output = pipeline.run(Payload(frames))

            metadata_strongsort[name] = output[StrongSORTCacheBenchmark]

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


# In[ ]:


# Stage.enable_progress()


# In[ ]:


if test == 'dev':
    test = 'true'

pipeline = Pipeline()

# Decode
pipeline.add_filter(DecodeFrame())
# 2D Detection
pipeline.add_filter(YoloDetection())
# Tracking
pipeline.add_filter(StrongSORTCacheBenchmark(cache=json.loads(test)))

for i in range(1):
    run_benchmark(pipeline, 'ss-cache-' + test, run=i, ignore_error=True)


# In[ ]:


if not is_notebook():
    subprocess.Popen('sudo shutdown -h now', shell=True)


# In[ ]:




