import json
import os
import pickle

from optimized_ingestion.camera_config import camera_config
from optimized_ingestion.payload import Payload
from optimized_ingestion.pipeline import Pipeline
from optimized_ingestion.stages.decode_frame.parallel_decode_frame import ParallelDecodeFrame
from optimized_ingestion.stages.decode_frame.decode_frame import DecodeFrame
from optimized_ingestion.stages.detection_2d.yolo_detection import YoloDetection
from optimized_ingestion.stages.filter_car_facing_sideway import FilterCarFacingSideway
from optimized_ingestion.stages.detection_estimation import DetectionEstimation
from optimized_ingestion.stages.tracking_2d.strongsort import StrongSORT
from optimized_ingestion.stages.tracking_2d.tracking_2d import Tracking2D
from optimized_ingestion.stages.tracking_3d.from_2d_and_road import From2DAndRoad
from optimized_ingestion.stages.tracking_3d.tracking_3d import Tracking3DResult
# from optimized_ingestion.trackers.yolov5_strongsort_osnet_tracker import TrackingResult
from optimized_ingestion.video import Video

BOSTON_VIDEOS = [
#     "scene-0757-CAM_FRONT",
    # "scene-0103-CAM_FRONT",
    # "scene-0553-CAM_FRONT",
    # "scene-0665-CAM_FRONT",
#     "scene-0655-CAM_FRONT_RIGHT",
    "scene-0655-CAM_BACK_RIGHT",
]

NUSCENES_PROCESSED_DATA = "NUSCENES_PROCESSED_DATA"


pipeline = Pipeline()
# pipeline.add_filter(filter=InView(distance=10, segment_type="intersection"))
# pipeline.add_filter(filter=Stopped(min_stopped_frames=2, stopped_threshold=1.0))
pipeline.add_filter(filter=DecodeFrame())
pipeline.add_filter(filter=YoloDetection())

# pipeline.add_filter(filter=DetectionEstimation())  # 5 Frame p Second
# pipeline.add_filter(filter=StrongSORT())  # 2 Frame p Second

# pipeline.add_filter(filter=From2DAndRoad())
# pipeline.add_filter(filter=FilterCarFacingSideway())

if NUSCENES_PROCESSED_DATA in os.environ:
    DATA_DIR = os.environ[NUSCENES_PROCESSED_DATA]
else:
    DATA_DIR = "/work/apperception/data/nuScenes/full-dataset-v1.0/Mini"
with open(os.path.join(DATA_DIR, "videos/boston-seaport", "frames.pickle"), "rb") as f:
    videos = pickle.load(f)

for name, video in videos.items():
#     if name not in BOSTON_VIDEOS:
#         continue
    # if not name.endswith('CAM_FRONT'):
    #     continue
        
#     if name != "scene-0553-CAM_FRONT":
#         continue

    print(name, '-----------------------------------------------------------------------------------------------------------------------------------------------------')
    frames = Video(
        os.path.join(DATA_DIR, "videos/boston-seaport", video["filename"]),
        [camera_config(*f, 0) for f in video["frames"]],
        video["start"],
    )

    output = pipeline.run(Payload(frames))

    # benchmark = []
    # for stage in pipeline.stages:
    #     benchmark.append({
    #         "stage": stage.classname(),
    #         "runtimes": stage.runtimes,
    #     })

    # with open("./outputs/benchmark.json", "w") as f3:
    #     json.dump(benchmark, f3)
