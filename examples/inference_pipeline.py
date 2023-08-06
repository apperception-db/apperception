import os
import pickle
from optimized_ingestion.actions.tracking2d_overlay import tracking2d_overlay

from optimized_ingestion.pipeline import Pipeline
from optimized_ingestion.payload import Payload
from optimized_ingestion.video import Video
from optimized_ingestion.camera_config import camera_config

from optimized_ingestion.stages.in_view.in_view import InView
from optimized_ingestion.stages.decode_frame.decode_frame import DecodeFrame
from optimized_ingestion.stages.detection_2d.yolo_detection import YoloDetection
from optimized_ingestion.stages.tracking_2d.strongsort import StrongSORT
from optimized_ingestion.stages.detection_2d.object_type_filter import ObjectTypeFilter
from optimized_ingestion.stages.detection_3d.from_detection_2d_and_road import FromDetection2DAndRoad

from optimized_ingestion.cache import disable_cache

disable_cache()

OUTPUT_DIR = './data/pipeline/test-results'
VIDEO_DIR =  './data/pipeline/videos'

files = os.listdir(VIDEO_DIR)

with open(os.path.join(VIDEO_DIR, 'frames.pkl'), 'rb') as f:
    videos = pickle.load(f)

pipeline = Pipeline([
    InView(50, 'intersection'),
    DecodeFrame(),
    YoloDetection(),
    ObjectTypeFilter(['car']),
    FromDetection2DAndRoad(),
    StrongSORT(),
])

for name, video in videos.items():
    if video['filename'] not in files:
        continue
    
    frames = Video(
        os.path.join(VIDEO_DIR, video["filename"]),
        [camera_config(*f, 0) for f in video["frames"]],
    )

    output = pipeline.run(Payload(frames))
    tracking2d_overlay(output, './examples/videos')
