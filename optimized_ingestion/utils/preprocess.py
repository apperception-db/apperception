from apperception.database import database
from apperception.utils import import_pickle

import json
import os
import pickle
import time

from optimized_ingestion.camera_config import camera_config
from optimized_ingestion.utils.process_pipeline import (construct_pipeline,
                                                        process_pipeline)
from optimized_ingestion.video import Video

BOSTON_VIDEOS = [
    "scene-0757-CAM_FRONT",
    # "scene-0103-CAM_FRONT",
    #     "scene-0553-CAM_FRONT",
    # "scene-0665-CAM_FRONT",
    # "scene-0655-CAM_FRONT",
    #     "scene-0655-CAM_FRONT_RIGHT",
    #     "scene-0655-CAM_BACK_RIGHT",
    #     "scene-0553-CAM_FRONT_LEFT"
    #     "scene-0103-CAM_FRONT"
]


def preprocess(world, data_dir, video_names=[], base=True, benchmark_path=None):
    pipeline = construct_pipeline(world, base=base)

    video_path = os.path.join(data_dir, "videos/")
    import_pickle(database, video_path)
    with open(os.path.join(video_path, 'frames.pkl'), "rb") as f:
        videos = pickle.load(f)

    if video_names:
        videos = {name: videos[name] for name in video_names}
    start_time = time.time()

    num_video = 0
    for name, video in videos.items():
        if video['location'] != 'boston-seaport':
            continue
        print(name, '--------------------------------------------------------------------------------')
        frames = Video(
            os.path.join(data_dir, "videos", video["filename"]),
            [camera_config(name, *f[1:], 0) for f in video["frames"]],
            video["start"],
        )
        process_pipeline(name, frames, pipeline, base)
        num_video += 1
    print(f"total preprocess time {time.time() - start_time}")

    if benchmark_path:
        total_runtime = 0
        stage_runtimes = []
        benchmarks = []
        for stage in pipeline.stages:
            stage_runtimes.append({
                "stage": stage.classname(),
                "runtimes": stage.runtimes,
            })
            total_runtime += sum([run['runtime'] for run in stage.runtimes])

        benchmarks.append({
            'stage_runtimes': stage_runtimes,
            'total_runtime': total_runtime
        })
        if num_video:
            benchmarks.append({'average runtime': sum([b['total_runtime'] for b in benchmarks]) / num_video})

        with open(benchmark_path, "w") as f3:
            json.dump(benchmarks, f3)
