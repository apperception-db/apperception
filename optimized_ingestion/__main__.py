import os
import pickle

from .camera_config import camera_config
from .payload import Payload
from .pipeline import Pipeline
from .utils.overlay_road import overlay_road
from .video import Video

"""
Query:
Find scenes where the ego car is 10 meters near an intersection
There is at least one car on an intersection driving in a ~perpendicular direction to the ego car
"""


BOSTON_VIDEOS = [
    "scene-0757-CAM_FRONT",
    # "scene-0103-CAM_FRONT",
    # "scene-0553-CAM_FRONT",
    # "scene-0665-CAM_FRONT",
]


if __name__ == "__main__":
    pipeline = Pipeline()
    # pipeline \
    #     .add_filter(filter=InView(distance=10, segment_type="intersection")) \
    #     .add_filter(filter=Stopped(min_stopped_frames=2, stopped_threshold=1.0)) \
    #     .add_filter(filter=DecodeFrame()) \
    #     .add_filter(filter=DepthEstimation()) \
    #     .add_filter(filter=Tracking2D()) \
    #     .add_filter(filter=From2DAndDepth()) \
    #     .add_filter(filter=FilterCarFacingSideway())

    if "NUSCENE_DATA" in os.environ:
        DATA_DIR = os.environ["NUSCENE_DATA"]
    else:
        DATA_DIR = "/work/apperception/data/nuScenes/full-dataset-v1.0/Mini"
    with open(os.path.join(DATA_DIR, "videos/boston-seaport", "frames.pickle"), "rb") as f:
        videos = pickle.load(f)

    for name, video in videos.items():
        if name not in BOSTON_VIDEOS:
            continue

        print(name)
        frames = Video(
            os.path.join(DATA_DIR, "videos/boston-seaport", video["filename"]),
            [camera_config(*f, 0) for f in video["frames"]],
            video["start"],
        )

        output = pipeline.run(Payload(frames))
        p = overlay_road(output)
        print(p[0][0])
        # output.save(f"./outputs/{name}.mp4")
