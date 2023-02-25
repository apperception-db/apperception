import os
import pickle

from apperception.database import database
from apperception.utils import import_pickle
from optimized_ingestion.camera_config import camera_config
from optimized_ingestion.utils.process_pipeline import construct_pipeline, process_pipeline
from optimized_ingestion.video import Video


BOSTON_VIDEOS = [
#     "scene-0757-CAM_FRONT",
    # "scene-0103-CAM_FRONT",
#     "scene-0553-CAM_FRONT",
    # "scene-0665-CAM_FRONT",
    "scene-0655-CAM_FRONT",
#     "scene-0655-CAM_FRONT_RIGHT",
#     "scene-0655-CAM_BACK_RIGHT",
#     "scene-0553-CAM_FRONT_LEFT"
#     "scene-0103-CAM_FRONT"
]
NUSCENES_PROCESSED_DATA = "NUSCENES_PROCESSED_DATA"

PICKLE_DATA_PATH = '/data/apperception-data/processed/nuscenes/full-dataset-v1.0/Mini/videos/boston-seaport'


def insert_camera():
    ### INSERT CAMERA
    import_pickle(database, PICKLE_DATA_PATH)

def preprocess(world, base=True):
    insert_camera()
    pipeline = construct_pipeline(world, base=base)
    if NUSCENES_PROCESSED_DATA in os.environ:
        DATA_DIR = os.environ[NUSCENES_PROCESSED_DATA]
    else:
        DATA_DIR = "/work/apperception/data/nuScenes/full-dataset-v1.0/Mini"
    with open(os.path.join(DATA_DIR, "videos/boston-seaport", "frames.pickle"), "rb") as f:
        videos = pickle.load(f)

    for name, video in videos.items():
        if name not in BOSTON_VIDEOS:
            continue
    #     if not name.endswith('CAM_FRONT'):
    #         continue
#         if 'CAM_FRONT' not in name:
#             continue

        print(name, '--------------------------------------------------------------------------------')
        frames = Video(
            os.path.join(DATA_DIR, "videos/boston-seaport", video["filename"]),
            [camera_config(*f, 0) for f in video["frames"]],
            video["start"],
        )
        process_pipeline(name, frames, pipeline, base)