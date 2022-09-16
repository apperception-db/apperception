import os
import pickle
from pathlib import Path

from filters import InViewFilter, TrackingFilter
from frame import frame
from frame_collection import FrameCollection
from payload import Payload
from pipeline import Pipeline


### Constants ###
SAMPLING_RATE = 2
CAMERA_ID = "scene-0757"
TEST_FILE_REG = '%CAM_FRONT/%2018-08-30-15%'
BASE_DIR = Path(__file__).resolve().parent.parent
TEST_FILE_DIR = os.path.join(BASE_DIR, 'data/v1.0-mini/')


if __name__ == "__main__":
    if 'NUSCENE_DATA' in os.environ:
        DATA_DIR = os.environ['NUSCENE_DATA']
    else:
        DATA_DIR = "/work/apperception/data/nuScenes/full-dataset-v1.0/Mini"
    with open(os.path.join(DATA_DIR, 'videos', 'frames.pickle'), "rb") as f:
        videos = pickle.load(f)

    # print([_ for _ in videos])
    video_0757 = videos['scene-0757-CAM_FRONT']
    frames = FrameCollection(
        os.path.join(DATA_DIR, 'videos', video_0757['filename']),
        [frame(*f) for f in video_0757['frames']],
        video_0757['start'],
    )
    pipeline = Pipeline()

    pipeline.add_filter(filter=InViewFilter(distance=10, segment_type="intersection"))
    pipeline.add_filter(filter=TrackingFilter())

    output = pipeline.run(Payload(frames))
