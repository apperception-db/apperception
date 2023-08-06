import json
import os
import pickle
import time

from spatialyze.database import database
from spatialyze.utils import import_pickle

from ..camera_config import camera_config
from ..utils.process_pipeline import construct_pipeline, process_pipeline
from ..video import Video


def preprocess(world, data_dir, video_names=[], base=True, insert_traj=True, benchmark_path=None):
    pipeline = construct_pipeline(world, base=base)

    video_path = os.path.join(data_dir, "videos/")
    import_pickle(database, video_path)
    with open(os.path.join(video_path, "frames.pkl"), "rb") as f:
        videos = pickle.load(f)

    if video_names:
        videos = {name: videos[name] for name in video_names}
    start_time = time.time()

    num_video = 0
    for name, video in videos.items():
        if video["location"] != "boston-seaport":
            continue
        if "FRONT" not in name:
            continue
        print(
            name, "--------------------------------------------------------------------------------"
        )
        frames = Video(
            os.path.join(data_dir, "videos", video["filename"]),
            [camera_config(name, *f[1:], 0) for f in video["frames"]],
            video["start"],
        )

        process_pipeline(name, frames, pipeline, base, insert_traj)
        num_video += 1

    print("num_video: ", num_video)

    print(f"total preprocess time {time.time() - start_time}")

    if benchmark_path:
        total_runtime = 0
        stage_runtimes = []
        benchmarks = []
        for stage in pipeline.stages:
            stage_runtimes.append(
                {
                    "stage": stage.classname(),
                    "runtimes": stage.benchmark,
                }
            )
            total_runtime += sum([run["runtime"] for run in stage.benchmark])

        benchmarks.append({"stage_runtimes": stage_runtimes, "total_runtime": total_runtime})
        if num_video:
            benchmarks.append(
                {"average runtime": sum([b["total_runtime"] for b in benchmarks]) / num_video}
            )
            benchmarks.append({"number of videos": num_video})

        with open(benchmark_path, "w") as f3:
            json.dump(benchmarks, f3)
