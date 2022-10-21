import dataclasses
import json
import os
import pickle
from dataclasses import is_dataclass

from .camera_config import camera_config
from .payload import Payload
from .pipeline import Pipeline
from .stages.decode_frame import DecodeFrame
from .stages.depth_estimation import DepthEstimation
from .stages.filter_car_facing_sideway import FilterCarFacingSideway
from .stages.in_view import InView
from .stages.stopped import Stopped
from .stages.tracking_2d import Tracking2D
from .stages.tracking_3d.from_2d_and_depth import From2DAndDepth
from .stages.tracking_3d.from_2d_and_road import From2DAndRoad
from .stages.tracking_3d.from_2d_and_road_naive import From2DAndRoadNaive
from .stages.tracking_3d.tracking_3d import Tracking3DResult
from .trackers.yolov5_strongsort_osnet_tracker import TrackingResult
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

NUSCENES_PROCESSED_DATA = "NUSCENES_PROCESSED_DATA"


class DataclassJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, Tracking3DResult):
            return {
                "object_id": o.object_id,
                "point_from_camera": o.point_from_camera,
                "point": o.point.tolist()
            }
        if isinstance(o, TrackingResult):
            return {
                "frame_idx": o.frame_idx,
                "object_id": o.object_id,
                "bbox_left": o.bbox_left,
                "bbox_top": o.bbox_top,
                "bbox_w": o.bbox_w,
                "bbox_h": o.bbox_h,
                "pred_idx": o.pred_idx,
                "object_type": o.object_type,
                "confidence": o.confidence
            }
        return super().default(o)


if __name__ == "__main__":
    pipeline = Pipeline()
    # pipeline \
    #     .add_filter(filter=InView(distance=10, segment_type="intersection")) \
    #     .add_filter(filter=Stopped(min_stopped_frames=2, stopped_threshold=1.0))
    pipeline \
        .add_filter(filter=DecodeFrame()) \
        .add_filter(filter=Tracking2D())
    pipeline \
        .add_filter(filter=From2DAndRoad())
    # # pipeline \
    #     .add_filter(filter=DepthEstimation()) \
    #     .add_filter(filter=From2DAndDepth())

    pipeline.add_filter(filter=FilterCarFacingSideway())

    if NUSCENES_PROCESSED_DATA in os.environ:
        DATA_DIR = os.environ[NUSCENES_PROCESSED_DATA]
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
        output.save(f"./outputs/{name}.mp4")

        # with open("./tracking2d.json", "w") as f2:
        #     tracking = Tracking2D.get(output.metadata)
        #     json.dump(tracking, f2, cls=DataclassJSONEncoder)

        # with open("./tracking3d.json", "w") as f2:
        #     tracking = From2DAndRoad.get(output.metadata)
        #     json.dump(tracking, f2, cls=DataclassJSONEncoder)

        # with open("./tracking3dnaive.json", "w") as f2:
        #     tracking = From2DAndRoadNaive.get(output.metadata)
        #     json.dump(tracking, f2, cls=DataclassJSONEncoder)

        # with open("./tracking3ddepth.json", "w") as f2:
        #     tracking = From2DAndDepth.get(output.metadata)
        #     json.dump(tracking, f2, cls=DataclassJSONEncoder)

        benchmark = []
        for stage in pipeline.stages:
            benchmark.append({
                "stage": stage.classname(),
                "runtimes": stage.runtimes,
            })

        with open("./benchmark.json", "w") as f3:
            json.dump(benchmark, f3)
