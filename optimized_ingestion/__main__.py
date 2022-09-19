import os
import pickle
from pathlib import Path

from .camera_config import camera_config
from .payload import Payload
from .pipeline import Pipeline
from .stages import InView, TrackingFilter
from .video import Video

### Constants ###
SAMPLING_RATE = 2
CAMERA_ID = "scene-0757"
TEST_FILE_REG = "%CAM_FRONT/%2018-08-30-15%"
BASE_DIR = Path(__file__).resolve().parent.parent
TEST_FILE_DIR = os.path.join(BASE_DIR, "data/v1.0-mini/")


if __name__ == "__main__":
    if "NUSCENE_DATA" in os.environ:
        DATA_DIR = os.environ["NUSCENE_DATA"]
    else:
        DATA_DIR = "/work/apperception/data/nuScenes/full-dataset-v1.0/Mini"
    with open(os.path.join(DATA_DIR, "videos", "frames.pickle"), "rb") as f:
        videos = pickle.load(f)

    video_0757 = videos["scene-0757-CAM_FRONT"]
    frames = Video(
        os.path.join(DATA_DIR, "videos", video_0757["filename"]),
        [camera_config(*f) for f in video_0757["frames"]],
        video_0757["start"],
    )
    pipeline = Pipeline()

    pipeline.add_filter(filter=InView(distance=10, segment_type="intersection"))
    pipeline.add_filter(filter=TrackingFilter())

    output = pipeline.run(Payload(frames))
