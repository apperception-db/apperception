import os
import pickle
import numpy as np
import json

from optimized_ingestion.pipeline import Pipeline
from optimized_ingestion.payload import Payload
from optimized_ingestion.video import Video
from optimized_ingestion.camera_config import camera_config

from optimized_ingestion.stages.decode_frame.decode_frame import DecodeFrame
from optimized_ingestion.stages.detection_2d.yolo_detection import YoloDetection
from optimized_ingestion.stages.detection_3d.from_detection_2d_and_road import FromDetection2DAndRoad

OUTPUT_DIR = './data/pipeline/outputs'
VIDEO_DIR =  './data/pipeline/videos'

def test_detection_3d():
    files = os.listdir(VIDEO_DIR)

    with open(os.path.join(VIDEO_DIR, 'frames.pkl'), 'rb') as f:
        videos = pickle.load(f)
    
    pipeline = Pipeline([
        DecodeFrame(),
        YoloDetection(),
        FromDetection2DAndRoad(),
    ])

    for name, video in videos.items():
        if video['filename'] not in files:
            continue
        
        frames = Video(
            os.path.join(VIDEO_DIR, video["filename"]),
            [camera_config(*f, 0) for f in video["frames"]],
        )

        output = pipeline.run(Payload(frames))
        det_result = FromDetection2DAndRoad.get(output)
        assert det_result is not None

        # with open(os.path.join(OUTPUT_DIR, f'FromDetection2DAndRoad--{name}.json'), 'w') as f:
        #     json.dump([(d[0].cpu().numpy().tolist(), d[1], d[2]) for d in det_result], f, indent=1)

        with open(os.path.join(OUTPUT_DIR, f'FromDetection2DAndRoad--{name}.json'), 'r') as f:
            det_groundtruth = json.load(f)
        
        for (det0, _, did0), (det1, _, did1) in zip(det_result, det_groundtruth):
            assert np.allclose(det0.cpu().numpy(), np.array(det1))
            assert np.allclose(np.array(did0), np.array(did1))
