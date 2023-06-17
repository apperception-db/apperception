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
from optimized_ingestion.stages.tracking_2d.strongsort import StrongSORT

OUTPUT_DIR = './data/pipeline/test-results'
VIDEO_DIR =  './data/pipeline/videos'

def test_detection_3d():
    files = os.listdir(VIDEO_DIR)

    with open(os.path.join(VIDEO_DIR, 'frames.pkl'), 'rb') as f:
        videos = pickle.load(f)
    
    pipeline = Pipeline([
        DecodeFrame(),
        YoloDetection(),
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
        track_result = StrongSORT.get(output)
        assert track_result is not None

        # with open(os.path.join(OUTPUT_DIR, f'StrongSORT--{name}.json'), 'w') as f:
        #     json.dump(track_result, f, indent=1, cls=MetadataJSONEncoder)

        with open(os.path.join(OUTPUT_DIR, f'StrongSORT--{name}.json'), 'r') as f:
            track_groundtruth = json.load(f)
        
        for tr, tg in zip(track_result, track_groundtruth):
            for oid, detr in tr.items():
                detg = tg[str(oid)]

                assert detr.frame_idx == detg['frame_idx']
                assert detr.object_id == detg['object_id']
                assert tuple(detr.detection_id) == tuple(detg['detection_id'])
                assert detr.bbox_left == detg['bbox_left']
                assert detr.bbox_top == detg['bbox_top']
                assert detr.bbox_w == detg['bbox_w']
                assert detr.bbox_h == detg['bbox_h']
                assert detr.object_type == detg['object_type']
                assert np.allclose(np.array([float(detr.confidence)]), np.array([detg['confidence']]))
