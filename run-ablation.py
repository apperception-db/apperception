#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import subprocess
import json
import os
import pickle
import traceback
import shutil
import socket
import time
import random

from os import environ

import numpy as np
import torch
import psycopg2

subprocess.Popen('nvidia-smi', shell=True).wait()
process = subprocess.Popen('docker container start mobilitydb', shell=True)


# In[ ]:


hostname = socket.gethostname()
test = hostname.split("-")[-1]
print("test", test)


# In[ ]:


def is_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            # Jupyter notebook or qtconsole
            return True
        elif shell == 'TerminalInteractiveShell':
            # Terminal running IPython
            return False
        else:
            # Other type (?)
            return False
    except NameError:
        # Probably standard Python interpreter
        return False


if is_notebook():
    get_ipython().run_line_magic('cd', '..')
    from tqdm.notebook import tqdm
    from nbutils.report_progress import report_progress
else:
    from tqdm import tqdm
    from playground.nbutils.report_progress import report_progress


# In[ ]:


process.wait()


# In[ ]:


from optimized_ingestion.camera_config import camera_config
from optimized_ingestion.payload import Payload
from optimized_ingestion.pipeline import Pipeline
from optimized_ingestion.video import Video
from optimized_ingestion.metadata_json_encoder import MetadataJSONEncoder


# In[ ]:


# Stages
from optimized_ingestion.stages.in_view import InView

from optimized_ingestion.stages.decode_frame.decode_frame import DecodeFrame

from optimized_ingestion.stages.detection_2d.detection_2d import Detection2D
from optimized_ingestion.stages.detection_2d.yolo_detection import YoloDetection
from optimized_ingestion.stages.detection_2d.object_type_filter import ObjectTypeFilter
from optimized_ingestion.stages.detection_2d.ground_truth import GroundTruthDetection


# In[ ]:


from optimized_ingestion.stages.detection_3d.from_detection_2d_and_road import FromDetection2DAndRoad
from optimized_ingestion.stages.detection_3d.from_detection_2d_and_depth import FromDetection2DAndDepth

from optimized_ingestion.stages.depth_estimation import DepthEstimation

from optimized_ingestion.stages.detection_estimation import DetectionEstimation
from optimized_ingestion.stages.detection_estimation.segment_mapping import RoadPolygonInfo


# In[ ]:


from optimized_ingestion.stages.tracking.strongsort import StrongSORT
from optimized_ingestion.stages.tracking_2d.strongsort import StrongSORT as StrongSORT2D


# In[ ]:


from optimized_ingestion.stages.tracking_3d.from_tracking_2d_and_road import FromTracking2DAndRoad
from optimized_ingestion.stages.tracking_3d.from_tracking_2d_and_depth import FromTracking2DAndDepth
from optimized_ingestion.stages.tracking_3d.tracking_3d import Tracking3DResult, Tracking3D

from optimized_ingestion.stages.segment_trajectory import SegmentTrajectory
from optimized_ingestion.stages.segment_trajectory.construct_segment_trajectory import SegmentPoint
from optimized_ingestion.stages.segment_trajectory.from_tracking_3d import FromTracking3D


# In[ ]:


from optimized_ingestion.cache import disable_cache
disable_cache()


# In[ ]:


from optimized_ingestion.utils.process_pipeline import format_trajectory, insert_trajectory, get_tracks
from optimized_ingestion.actions.tracking2d_overlay import tracking2d_overlay


# In[ ]:


from apperception.utils.ingest_road import ingest_road
from apperception.database import database, Database
from apperception.world import empty_world
from apperception.utils import F
from apperception.predicate import camera, objects, lit, FindAllTablesVisitor, normalize, MapTablesTransformer, GenSqlVisitor
from apperception.data_types.camera import Camera as ACamera
from apperception.data_types.camera_config import CameraConfig as ACameraConfig


# In[ ]:


NUSCENES_PROCESSED_DATA = "NUSCENES_PROCESSED_DATA"
print(NUSCENES_PROCESSED_DATA in os.environ)
print(os.environ['NUSCENES_PROCESSED_DATA'])


# In[ ]:


DATA_DIR = os.environ[NUSCENES_PROCESSED_DATA]
# with open(os.path.join(DATA_DIR, "videos", "frames.pkl"), "rb") as f:
#     videos = pickle.load(f)
with open(os.path.join(DATA_DIR, 'videos', 'videos.json'), 'r') as f:
    videos = json.load(f)


# In[ ]:


with open('./data/evaluation/video-samples/boston-seaport.txt', 'r') as f:
    sampled_scenes = f.read().split('\n')
print(sampled_scenes[0], sampled_scenes[-1], len(sampled_scenes))


# In[ ]:


BENCHMARK_DIR = "./outputs/run"


def bm_dir(*args: "str"):
    return os.path.join(BENCHMARK_DIR, *args)


# In[ ]:


def get_sql(predicate: "PredicateNode"):
    tables, camera = FindAllTablesVisitor()(predicate)
    tables = sorted(tables)
    mapping = {t: i for i, t in enumerate(tables)}
    predicate = normalize(predicate)
    predicate = MapTablesTransformer(mapping)(predicate)

    t_tables = ''
    t_outputs = ''
    for i in range(len(tables)):
        t_tables += '\n' \
            'JOIN Item_General_Trajectory ' \
            f'AS t{i} ' \
            f'ON Cameras.timestamp <@ t{i}.trajCentroids::period'
        t_outputs += f', t{i}.itemId'

    return f"""
        SELECT Cameras.frameNum {t_outputs}
        FROM Cameras{t_tables}
        WHERE
        {GenSqlVisitor()(predicate)}
    """


# In[ ]:


slices = {
    "noopt": (0, 15),
    "inview": (15, 30),
    "objectfilter": (30, 45),
    "geo": (45, 60),
    "de": (60, 75),
    "opt": (75, 90),
    "optde": (90, 105),
    'mick-dev': (0, 15),
    'freddie': (15, 30),
}


# In[ ]:


def run_benchmark(pipeline, filename, predicates, run=0, ignore_error=False):
    print(filename)
    metadata_strongsort = {}
    metadata_d2d = {}
    failed_videos = []
    runtime_input = []
    runtime_query = []
    runtime_video = []

    all_metadata = {
        'detection': metadata_d2d,
        'sort': metadata_strongsort,
    }
    print('# of total    videos:', len(videos))

    names = set(sampled_scenes[:35])
    # names = {'0655'}
    filtered_videos = [
        n for n in videos
        if n[6:10] in names and 'FRONT' in n
    ]
    print('# of filtered videos:', len(filtered_videos))

    s_from, s_to = slices[test]
    filtered_videos = filtered_videos[s_from:s_to]
    print('# of sliced   videos:', len(filtered_videos))
    # ingest_road(database, './data/scenic/road-network/boston-seaport')

    for pre in [*all_metadata.keys(), 'qresult', 'performance']:
        p = os.path.join(BENCHMARK_DIR, f"{pre}--{filename}_{run}")
        if os.path.exists(p):
            shutil.rmtree(p)
        os.makedirs(p)

    def save_perf():
        with open(bm_dir(f"failed_videos--{filename}_{run}.json"), "w") as f:
            json.dump(failed_videos, f, indent=1)

        with open(bm_dir(f"perf--{filename}_{run}.json"), "w") as f:
            performance = [
                {
                    "stage": stage.classname(),
                    "benchmark": stage.benchmark,
                    **(
                        {'explains': stage.explains}
                        if hasattr(stage, 'explains')
                        else {}
                    ),
                    **(
                        {"ss-benchmark": stage.ss_benchmark}
                        if hasattr(stage, 'ss_benchmark')
                        else {}
                    )
                }
                for stage
                in pipeline.stages
            ]
            json.dump(performance, f, indent=1)
        with open(bm_dir(f"perfexec--{filename}_{run}.json"), 'w') as f:
            json.dump({
                'ingest': 2.2629338979721068,
                'input': runtime_input,
                'query': runtime_query,
                'save': runtime_video
            }, f, indent=1)

    for i, name in tqdm(enumerate(filtered_videos), total=len(filtered_videos)):
        # if i % int(len(filtered_videos) / 200) == 0:
        #     report_progress(i, len(filtered_videos), filename, str(run))
        try:
            start_input = time.time()
            with open(os.path.join(DATA_DIR, 'videos', 'boston-seaport-' + name + '.pkl'), 'rb') as f:
                video = pickle.load(f)
            video_filename = video['filename']

            frames = Video(
                os.path.join(DATA_DIR, "videos", video["filename"]),
                [camera_config(*f, 0) for f in video["frames"]],
            )
            time_input = time.time() - start_input
            runtime_input.append({'name': name, 'runtime': time_input})

            output = pipeline.run(Payload(frames))

            metadata_strongsort[name] = output[StrongSORT2D]
            metadata_d2d[name] = output[Detection2D]

            for pre, metadata in all_metadata.items():
                p = bm_dir(f"{pre}--{filename}_{run}", f"{name}.json")
                with open(p, "w") as f:
                    json.dump(metadata[name], f, cls=MetadataJSONEncoder, indent=1)

            times_rquery = []
            for i, predicate in enumerate(predicates):
                start_rquery = time.time()
                database.reset(True)

                # Ingest Trackings
                ego_meta = frames.interpolated_frames
                sortmeta = FromTracking2DAndRoad.get(output)
                segment_trajectory_mapping = FromTracking3D.get(output)
                tracks = get_tracks(sortmeta, ego_meta, segment_trajectory_mapping, True)
                for obj_id, track in tracks.items():
                    trajectory = format_trajectory(name, obj_id, track, True)
                    if trajectory:
                        insert_trajectory(database, *trajectory)

                # Ingest Camera
                accs: 'ACameraConfig' = []
                for i, cc in enumerate(frames.interpolated_frames):
                    acc = ACameraConfig(
                        frame_id=cc.frame_id,
                        frame_num=i,
                        filename=cc.filename,
                        camera_translation=cc.camera_translation,
                        camera_rotation=cc.camera_rotation,
                        camera_intrinsic=cc.camera_intrinsic,
                        ego_translation=cc.ego_translation,
                        ego_rotation=cc.ego_rotation,
                        timestamp=cc.timestamp,
                        cameraHeading=cc.camera_heading,
                        egoHeading=cc.ego_heading,
                    )
                    accs.append(acc)
                camera = ACamera(accs, cc.camera_id)
                database.insert_cam(camera)

                query = get_sql(predicate)
                # t0 = time.time()
                qresult = database.execute(query)
                # print('query', time.time() - t0)
                # print('result length', len(qresult))
                # print(qresult)

                p = bm_dir(f"qresult--{filename}_{run}", f"{name}-{i}.json")
                with open(p, 'w') as f:
                    json.dump(qresult, f, indent=1)
                time_rquery = time.time() - start_rquery
                times_rquery.append(time_rquery)
                runtime_query.append({'name': name, 'predicate': i, 'runtime': time_rquery})
                # print()

            # save video
            start_video = time.time()
            tracking2d_overlay(output, './tmp.mp4')
            time_video = time.time() - start_video
            runtime_video.append({'name': name, 'runtime': time_video})

            perf = []
            for stage in pipeline.stages:
                benchmarks = [*filter(lambda x: video['filename'] in x['name'], stage.benchmark)]
                assert len(benchmarks) == 1
                perf.append({
                    'stage': stage.classname(),
                    'benchmark': benchmarks[0]
                })

            perf.append({
                'stage': 'ingest',
                'benchmark': {
                    'name': name,
                    'runtime': time_input
                }
            })
            perf.append({
                'stage': 'save',
                'benchmark': {
                    'name': name,
                    'runtime': time_video
                }
            })
            for i, time_rquery in enumerate(times_rquery):
                perf.append({
                    'stage': 'query',
                    'benchmark': {
                        'name': name,
                        'predicate': i,
                        'runtime': time_rquery
                    }
                })
            p = bm_dir(f'performance--{filename}_{run}', f'{name}.json')
            with open(p, "w") as f:
                json.dump(perf, f, indent=1)
        except Exception as e:
            if ignore_error:
                message = str(traceback.format_exc())
                failed_videos.append((name, message))
                print(video_filename)
                print(e)
                print(message)
                print("------------------------------------------------------------------------------------")
                print()
                print()
            else:
                raise e

        if len(metadata_d2d) % 10 == 0:
            save_perf()
    save_perf()


# In[ ]:


def create_pipeline(
    predicate,
    in_view=True,
    object_filter=True,
    groundtruth_detection=False,
    geo_depth=True,
    detection_estimation=True,
    strongsort=False,
    ss_update_when_skip=True,
):
    pipeline = Pipeline()

    # In-View Filter
    if in_view:
        # TODO: view angle and road type should depends on the predicate
        pipeline.add_filter(InView(50, predicate=predicate))

    # Decode
    pipeline.add_filter(DecodeFrame())

    # 2D Detection
    if groundtruth_detection:
        with open(os.path.join(DATA_DIR, 'annotation_partitioned.pkl'), 'rb') as f:
            df_annotations = pickle.load(f)
        pipeline.add_filter(GroundTruthDetection(df_annotations))
    else:
        pipeline.add_filter(YoloDetection())

    # Object Filter
    if object_filter:
        # if isinstance(object_filter, bool):
        #     object_filter = ['car', 'truck']
        # TODO: filter objects based on predicate
        pipeline.add_filter(ObjectTypeFilter(predicate=predicate))

    # 3D Detection
    if geo_depth:
        pipeline.add_filter(FromDetection2DAndRoad())
    else:
        pipeline.add_filter(DepthEstimation())
        pipeline.add_filter(FromDetection2DAndDepth())

    # Detection Estimation
    if detection_estimation:
        pipeline.add_filter(DetectionEstimation())

    # Tracking
    pipeline.add_filter(StrongSORT2D(
        # method='update-empty' if ss_update_when_skip else 'increment-ages',
        method='update-empty',
        cache=True
    ))

    pipeline.add_filter(FromTracking2DAndRoad())

    # Segment Trajectory
    pipeline.add_filter(FromTracking3D())

    return pipeline


# In[ ]:


predicate = None

p_noOpt = lambda predicate: create_pipeline(
    predicate,
    in_view=False,
    object_filter=False,
    geo_depth=False,
    detection_estimation=False
)

p_inview = lambda predicate: create_pipeline(
    predicate,
    in_view=True,
    object_filter=False,
    geo_depth=False,
    detection_estimation=False
)

p_objectFilter = lambda predicate: create_pipeline(
    predicate,
    in_view=False,
    object_filter=True,
    geo_depth=False,
    detection_estimation=False
)

p_geo = lambda predicate: create_pipeline(
    predicate,
    in_view=False,
    object_filter=False,
    geo_depth=True,
    detection_estimation=False
)

p_de = lambda predicate: create_pipeline(
    predicate,
    in_view=False,
    object_filter=False,
    geo_depth=False,
    detection_estimation=True
)

p_deIncr = lambda predicate: create_pipeline(
    predicate,
    in_view=False,
    object_filter=False,
    geo_depth=False,
    detection_estimation=True,
    ss_update_when_skip=False,
)

p_opt = lambda predicate: create_pipeline(
    predicate,
    in_view=True,
    object_filter=True,
    geo_depth=True,
    detection_estimation=False
)

p_optDe = lambda predicate: create_pipeline(
    predicate,
    in_view=True,
    object_filter=True,
    geo_depth=True,
    detection_estimation=True
)

p_optIncr = lambda predicate: create_pipeline(
    predicate,
    in_view=True,
    object_filter=True,
    geo_depth=True,
    detection_estimation=False,
    ss_update_when_skip=False,
)

p_optDeIncr = lambda predicate: create_pipeline(
    predicate,
    in_view=True,
    object_filter=True,
    geo_depth=True,
    detection_estimation=True,
    ss_update_when_skip=False,
)

p_gtOpt = lambda predicate: create_pipeline(
    predicate,
    in_view=True,
    object_filter=True,
    groundtruth_detection=True,
    geo_depth=True,
    detection_estimation=False
)

p_gtOptDe = lambda predicate: create_pipeline(
    predicate,
    in_view=True,
    object_filter=True,
    groundtruth_detection=True,
    geo_depth=True,
    detection_estimation=True
)

pipelines = {
    "noopt": p_noOpt,
    "inview": p_inview,
    "objectfilter": p_objectFilter,
    "geo": p_geo,
    "de": p_de,
    # "deincr": p_deIncr,
    "opt": p_opt,
    # "optincr": p_optIncr,
    "optde": p_optDe,
    # "optdeincr": p_optDeIncr,

    # "gtopt": p_gtOpt,
    # "gtoptde": p_gtOptDe
}


# In[ ]:


if test == 'dev':
    test = 'opt'


# In[ ]:


def run(test):
    o = objects[0]
    c = camera
    pred1 = (
        (o.type == 'person') &
        # F.contained(c.ego, 'intersection') r
        F.contained(o.trans@c.time, 'intersection') &
        # F.angle_excluding(F.facing_relative(o.traj@c.time, c.ego), lit(-70), lit(70)) &
        F.angle_between(F.facing_relative(c.cam, F.road_direction(c.ego)), lit(-15), lit(15)) &
        (F.distance(c.cam, o.traj@c.time) < lit(50)) # &
        # (F.view_angle(o.trans@c.time, c.camAbs) < lit(35))
    )

    obj1 = objects[0]
    obj2 = objects[1]
    cam = camera
    pred2 = (
        (obj1.id != obj2.id) &
        ((obj1.type == 'car') | (obj1.type == 'truck')) &
        ((obj2.type == 'car') | (obj2.type == 'truck')) &
        F.angle_between(F.facing_relative(cam.cam, F.road_direction(cam.cam)), -15, 15) &
        (F.distance(cam.cam, obj1.trans@cam.time) < 50) &
        # (F.view_angle(obj1.trans@cam.time, cam.ego) < 70 / 2.0) &
        (F.distance(cam.cam, obj2.trans@cam.time) < 50) &
        # (F.view_angle(obj2.trans@cam.time, cam.ego) < 70 / 2.0) &
        F.contains_all('intersection', [obj1.trans, obj2.trans]@cam.time)# &
        # F.angle_between(F.facing_relative(obj1.trans@cam.time, cam.ego), 40, 135) &
        # F.angle_between(F.facing_relative(obj2.trans@cam.time, cam.ego), -135, -50) &
        # (F.min_distance(cam.ego, 'intersection') < 10) &
        # F.angle_between(F.facing_relative(obj1.trans@cam.time, obj2.trans@cam.time), 100, -100)
    )

    obj1 = objects[0]
    cam = camera
    pred3 = (
        ((obj1.type == 'car') | (obj1.type == 'truck')) &
        (F.distance(cam.cam, obj1.trans@cam.timestamp) < 50) &
        # (F.view_angle(obj1.trans@cam.time, cam.ego) < 70 / 2) &
        F.angle_between(F.facing_relative(cam.cam, F.road_direction(cam.cam, cam.cam)), -180, -90) &
        F.contained(cam.cam, F.road_segment('road')) &
        F.contained(obj1.trans@cam.time, F.road_segment('road')) &
        # F.angle_between(F.facing_relative(obj1.trans@cam.time, F.road_direction(obj1.traj@cam.time, cam.ego)), -15, 15) &
        # F.angle_between(F.facing_relative(obj1.trans@cam.time, cam.ego), 135, 225) &
        (F.distance(cam.cam, obj1.trans@cam.time) < 10)
    )

    cam = camera
    car1 = objects[0]
    opposite_car = objects[1]
    car2 = objects[2]

    pred4 = (
        ((car1.type == 'car') | (car1.type == 'truck')) &
        ((car2.type == 'car') | (car2.type == 'truck')) &
        ((opposite_car.type == 'car') | (opposite_car.type == 'truck')) &
        (opposite_car.id != car2.id) &
        (car1.id != car2.id) &
        (car1.id != opposite_car.id) &

        F.angle_between(F.facing_relative(cam.cam, F.road_direction(cam.cam, cam.cam)), -15, 15) &
        # (F.view_angle(car1.traj@cam.time, cam.ego) < 70 / 2) &
        (F.distance(cam.cam, car1.traj@cam.time) < 40) &
        # F.angle_between(F.facing_relative(car1.traj@cam.time, cam.ego), -15, 15) &
        # F.angle_between(F.facing_relative(car1.traj@cam.time, F.road_direction(car1.traj@cam.time, cam.ego)), -15, 15) &
        F.ahead(car1.traj@cam.time, cam.cam) &
        # (F.convert_camera(opposite_car.traj@cam.time, cam.ego) > [-10, 0]) &
        # (F.convert_camera(opposite_car.traj@cam.time, cam.ego) < [-1, 50]) &
        # F.angle_between(F.facing_relative(opposite_car.traj@cam.time, cam.ego), 140, 180) &
        # (F.distance(opposite_car@cam.time, car2@cam.time) < 40)# &
        # F.angle_between(F.facing_relative(car2.traj@cam.time, F.road_direction(car2.traj@cam.time, cam.ego)), -15, 15) &
        F.ahead(car2.traj@cam.time, opposite_car.traj@cam.time)
    )

    p1 = pipelines[test](pred1)
    p2 = pipelines[test](pred2)
    p34 = pipelines[test](pred3)

    print(p2)
    run_benchmark(p2, 'q2-' + test, [pred2], run=1, ignore_error=True)

    print(p34)
    run_benchmark(p34, 'q34-' + test, [pred3, pred4], run=1, ignore_error=True)

    if test != 'optde' and test != 'de':
        print(p1)
        run_benchmark(p1, 'q1-' + test, [pred1], run=1, ignore_error=True)


# In[ ]:


tests = ['noopt', 'inview', 'objectfilter', 'geo', 'de', 'opt', 'optde']
random.shuffle(tests)

for test in tests:
    assert isinstance(pipelines[test](lit(True)), Pipeline)

for test in tests:
    run(test)


# In[ ]:


# run(test)


# In[ ]:


# if test == 'opt':
#     run('optde')


# In[ ]:


if not is_notebook():
    subprocess.Popen('sudo shutdown -h now', shell=True)


# In[ ]:




