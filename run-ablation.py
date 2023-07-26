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
    "noopt": (0, 1),
    "inview": (1, 2),
    "objectfilter": (2, 3),
    "geo": (3, 4),
    "de": (4, 5),
    "opt": (5, 6),
    # "optde": (6, 7),
    'dev': (0, 2),
    'freddie': (1, 2),
}


# In[ ]:


def run_benchmark(pipeline, filename, predicates, run=0, ignore_error=False):
    print(filename)
    metadata_strongsort = {}
    metadata_d2d = {}
    failed_videos = []

    all_metadata = {
        'detection': metadata_d2d,
        'sort': metadata_strongsort,
    }
    print('# of total    videos:', len(videos))

    names = set(sampled_scenes[:20])
    filtered_videos = [
        n for n in videos
        if n[6:10] in names
    ]
    N = len(filtered_videos)
    print('# of filtered videos:', N)

    # s_from, s_to = slices[test]
    s_from, s_to = (int(test), int(test) + 1)
    STEP = N // 5
    print('test', test)
    print('from', s_from*STEP)
    print('to  ', s_to*STEP)
    filtered_videos = filtered_videos[s_from*STEP:s_to*STEP]
    print('# of sliced   videos:', len(filtered_videos))
    # ingest_road(database, './data/scenic/road-network/boston-seaport')

    for pre in [*all_metadata.keys(), 'qresult', 'performance', 'failedvideos']:
        p = os.path.join(BENCHMARK_DIR, f"{pre}--{filename}_{run}")
        if os.path.exists(p):
            shutil.rmtree(p)
        os.makedirs(p)

    def save_perf():
        for n, message in failed_videos:
            p = bm_dir(f'failedvideos--{filename}_{run}', f'{n}.txt')
            with open(p, "w") as f:
                f.write(message)

    for i, name in tqdm(enumerate(filtered_videos), total=len(filtered_videos)):
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
                for idx, cc in enumerate(frames.interpolated_frames):
                    acc = ACameraConfig(
                        frame_id=cc.frame_id,
                        frame_num=idx,
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
                qresult = database.execute(query)

                p = bm_dir(f"qresult--{filename}_{run}", f"{name}-{i}.json")
                with open(p, 'w') as f:
                    json.dump(qresult, f, indent=1)
                time_rquery = time.time() - start_rquery
                times_rquery.append(time_rquery)
                # runtime_query.append({'name': name, 'predicate': i, 'runtime': time_rquery})

            # save video
            start_video = time.time()
            tracking2d_overlay(output, './tmp.mp4')
            time_video = time.time() - start_video
            # runtime_video.append({'name': name, 'runtime': time_video})

            perf = []
            for stage in pipeline.stages:
                benchmarks = [*filter(
                    lambda x: video['filename'] in x['name'],
                    stage.benchmark
                )]
                assert len(benchmarks) == 1
                perf.append({
                    'stage': stage.classname(),
                    'benchmark': benchmarks[0]
                })

                for bm in getattr(stage, '_benchmark', []):
                    if video['filename'] in bm['name']:
                        perf.append({
                            'stage': stage.classname(),
                            'addition': True,
                            'benchmark': bm,
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
    ss_cache=True,
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
        cache=ss_cache,
    ))

    pipeline.add_filter(FromTracking2DAndRoad())

    # Segment Trajectory
    pipeline.add_filter(FromTracking3D())

    return pipeline


# In[ ]:


p_noSSOpt = lambda predicate: create_pipeline(
    predicate,
    in_view=False,
    object_filter=False,
    geo_depth=False,
    detection_estimation=False,
    ss_cache=False,
)

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
    "nossopt": p_noSSOpt,
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


# if test == 'dev':
#     test = 'opt'


# In[ ]:


def run(__test):
    o = objects[0]
    c = camera
    pred1 = (
        (o.type == 'person') &
        # F.contained(c.ego, 'intersection') &
        F.contained(o.trans@c.time, 'intersection') &
        F.angle_excluding(F.facing_relative(o.traj@c.time, c.cam), lit(-70), lit(70)) &
        # F.angle_between(F.facing_relative(c.cam, F.road_direction(c.ego)), lit(-15), lit(15)) &
        (F.distance(c.cam, o.traj@c.time) < lit(50)) # &
        # (F.view_angle(o.trans@c.time, c.camAbs) < lit(35))
    )
    pred1_notrack = (
        F.contained(o.trans@c.time, 'intersection') &
        (F.distance(c.cam, o.traj@c.time) < lit(50)) &
        (o.type == 'person')
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
        F.contains_all('intersection', [obj1.trans, obj2.trans]@cam.time) &
        # F.angle_between(F.facing_relative(obj1.trans@cam.time, cam.cam), 40, 135) &
        # F.angle_between(F.facing_relative(obj2.trans@cam.time, cam.cam), -135, -50)
        F.angle_between(F.facing_relative(obj1.trans@cam.time, obj2.trans@cam.time), -180, -90)
        # (F.min_distance(cam.ego, 'intersection') < 10) &
        # F.angle_between(F.facing_relative(obj1.trans@cam.time, obj2.trans@cam.time), 100, -100)
    )
    pred2_notrack = (
        (F.distance(cam.cam, obj1.trans@cam.time) < 50) &
        (F.distance(cam.cam, obj2.trans@cam.time) < 50) &
        F.contains_all('intersection', [obj1.trans, obj2.trans]@cam.time) &
        (obj1.id != obj2.id) &
        ((obj1.type == 'car') | (obj1.type == 'truck')) &
        ((obj2.type == 'car') | (obj2.type == 'truck'))
    )

    obj1 = objects[0]
    cam = camera
    pred3 = (
        ((obj1.type == 'car') | (obj1.type == 'truck')) &
        # (F.view_angle(obj1.trans@cam.time, cam.ego) < 70 / 2) &
        F.angle_between(F.facing_relative(cam.cam, F.road_direction(cam.ego, cam.ego)), 135, 225) &
        F.contained(cam.cam, F.road_segment('lane')) &
        F.contained(obj1.trans@cam.time, F.road_segment('lane')) &
        F.angle_between(F.facing_relative(obj1.trans@cam.time, F.road_direction(obj1.traj@cam.time, cam.ego)), -15, 15) &
        # F.angle_between(F.facing_relative(obj1.trans@cam.time, cam.ego), 135, 225) &
        (F.distance(cam.cam, obj1.trans@cam.time) < 10)
    )
    pred3_notrack = (
        # F.contained(cam.cam, F.road_segment('lane')) &
        F.contained(obj1.trans@cam.time, F.road_segment('lane')) &
        (F.distance(cam.cam, obj1.trans@cam.time) < 10) &
        ((obj1.type == 'car') | (obj1.type == 'truck'))
    )

    cam = camera
    car1 = objects[0]
    opposite_car_1 = objects[1]
    opposite_car_2 = objects[2]

    pred4 = (
        ((car1.type == 'car') | (car1.type == 'truck')) &
        ((opposite_car_2.type == 'car') | (opposite_car_2.type == 'truck')) &
        ((opposite_car_1.type == 'car') | (opposite_car_1.type == 'truck')) &
        (opposite_car_1.id != opposite_car_2.id) &
        (car1.id != opposite_car_2.id) &
        (car1.id != opposite_car_1.id) &

        F.contained(cam.cam, F.road_segment('lane')) &
        F.contained(car1.trans@cam.time, F.road_segment('lane')) &
        F.contained(opposite_car_1.trans@cam.time, F.road_segment('lane')) &
        F.contained(opposite_car_2.trans@cam.time, F.road_segment('lane')) &
        F.angle_between(F.facing_relative(cam.cam, F.road_direction(cam.cam, cam.cam)), -15, 15) &
        # (F.view_angle(car1.traj@cam.time, cam.ego) < 70 / 2) &
        (F.distance(cam.cam, car1.traj@cam.time) < 40) &
        F.angle_between(F.facing_relative(car1.traj@cam.time, cam.ego), -15, 15) &
        # F.angle_between(F.facing_relative(car1.traj@cam.time, F.road_direction(car1.traj@cam.time, cam.ego)), -15, 15) &
        F.ahead(car1.traj@cam.time, cam.cam) &
        # (F.convert_camera(opposite_car.traj@cam.time, cam.ego) > [-10, 0]) &
        # (F.convert_camera(opposite_car.traj@cam.time, cam.ego) < [-1, 50]) &
        F.angle_between(F.facing_relative(opposite_car_1.traj@cam.time, cam.ego), 135, 225) &
        # (F.distance(opposite_car@cam.time, car2@cam.time) < 40)# &
        F.angle_between(F.facing_relative(opposite_car_2.traj@cam.time, opposite_car_1.traj@cam.time), -15, 15) &
        F.angle_between(F.facing_relative(opposite_car_2.traj@cam.time, F.road_direction(opposite_car_2.traj@cam.time, cam.ego)), -15, 15) &
        F.ahead(opposite_car_2.traj@cam.time, opposite_car_1.traj@cam.time)
    )
    pred4_notrack = (
        # F.contained(cam.cam,                       F.road_segment('lane')) &
        F.contained(car1.trans@cam.time,           F.road_segment('lane')) &
        F.contained(opposite_car_1.trans@cam.time, F.road_segment('lane')) &
        F.contained(opposite_car_2.trans@cam.time, F.road_segment('lane')) &
        ((car1.type == 'car') | (car1.type == 'truck')) &
        ((opposite_car_2.type == 'car') | (opposite_car_2.type == 'truck')) &
        ((opposite_car_1.type == 'car') | (opposite_car_1.type == 'truck')) &
        (opposite_car_1.id != opposite_car_2.id) &
        (car1.id != opposite_car_2.id) &
        (car1.id != opposite_car_1.id)
    )

    p1 = pipelines[__test](pred1)
    p2 = pipelines[__test](pred2)
    p34 = pipelines[__test](pred3)

    print(p2)
    run_benchmark(p2, 'q2-' + __test, [pred2, pred2_notrack], run=1, ignore_error=True)

    print(p34)
    run_benchmark(p34, 'q34-' + __test, [pred3, pred4, pred3_notrack, pred4_notrack], run=1, ignore_error=True)

    if __test != 'optde' and __test != 'de':
        print(p1)
        run_benchmark(p1, 'q1-' + __test, [pred1, pred1_notrack], run=1, ignore_error=True)


# In[ ]:


tests = ['de', 'optde', 'noopt', 'inview', 'objectfilter', 'geo', 'opt']
# tests = ['de', 'optde']
# random.shuffle(tests)

for _test in tests:
    assert isinstance(pipelines[_test](lit(True)), Pipeline)

for idx, _test in enumerate(tests):
    print(f'----------- {idx} / {len(tests)} --- {_test} -----------')
    run(_test)


# In[ ]:


# run(test)


# In[ ]:


# if test == 'opt':
#     run('optde')


# In[ ]:


if not is_notebook():
    subprocess.Popen('sudo shutdown -h now', shell=True)


# In[ ]:




