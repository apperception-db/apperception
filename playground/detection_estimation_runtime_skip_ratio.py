import os
os.chdir("..")
import time
import json

from apperception.database import database
from apperception.world import empty_world
from apperception.utils import F
from apperception.predicate import camera, objects
from optimized_ingestion.utils.preprocess import preprocess
database.connection
from optimized_ingestion.cache import disable_cache
disable_cache()

NUSCENES_PROCESSED_DATA = "NUSCENES_PROCESSED_DATA"
if NUSCENES_PROCESSED_DATA in os.environ:
    DATA_DIR = os.environ[NUSCENES_PROCESSED_DATA]
else:
    DATA_DIR = "/data/processed/full-dataset/trainval"
NUSCENES_RAW_DATA = "NUSCENES_RAW_DATA"
if NUSCENES_RAW_DATA in os.environ:
    RAW_DATA_DIR = os.environ[NUSCENES_RAW_DATA]
else:
    RAW_DATA_DIR = "/data/full-dataset/trainval"
    
with open('data/evaluation/video-samples/boston-seaport.txt', 'r') as f:
    scenes = f.read().split('\n')
    
def bechmark_detection_estimation_gain(world,video_names=[], scenes=[], path_suffix=None, start_ratio=1, end_ratio=10):
    base_benchmark_path = f'./outputs/base_pipeline_benchmark{"_"+path_suffix if path_suffix else ""}.json'
    ### base pipeline
    preprocess(world, DATA_DIR, video_names, scenes, insert_traj=False,
               benchmark_path=base_benchmark_path)
    ### detection estimation benchmark
    for skip_ratio in [0.1 * i for i in range(start_ratio,end_ratio)]:
        print(f"current skip ratio{skip_ratio}")
        optimize_benchmark_path = f'./outputs/detection_estimation_pipeline_{int(skip_ratio*10)}{"_"+path_suffix if path_suffix else ""}.json'
        preprocess(world, DATA_DIR, video_names, scenes,
                   base=False,
                   benchmark_path=optimize_benchmark_path,
                   insert_traj=False,
                   skip_ratio=skip_ratio)
        
name = 'ScenicWorld' # world name
world = empty_world(name=name)

bechmark_detection_estimation_gain(world, scenes=scenes[:200])

# obj1 = objects[0]
# cam = camera
# car_world = empty_world(name=name).filter(
#     (F.like(obj1.type, 'car') | F.like(obj1.type, 'truck') | F.like(obj1.type, 'bus'))
# )

# bechmark_detection_estimation_gain(car_world, scenes=scenes[:200], path_suffix="only_car", start_ratio=9)