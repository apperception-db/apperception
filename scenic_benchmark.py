import os
import sys
sys.path.append(os.path.join(os.getcwd(),"apperception"))

### IMPORTS
import cv2

from scenic_world import *
from world_executor import *
from video_util import *
from metadata_util import *
import json
import pandas as pd

from scenic_generate_df import *

### Let's define some attribute for constructing the world first
name = 'ScenicWorld' # world name
units = 'metrics'      # world units
user_data_dir = os.path.join("v1.0-mini")

# generate dataframes from scenic json files
# takes 1min to run
df_sample_data, df_annotation = scenic_generate_df()

scenic_world = ScenicWorld(name=name, units=units)

### Ingest the camera to the world
scene_name = "scene-0103"
scenic_world_61 = scenic_world.scenic_camera(scene_name)

### Call execute on the world to run the detection algorithm and save the real data to the database
recognized_scenic_world = scenic_world_61.recognize(scene_name, df_sample_data, df_annotation)
recognized_scenic_world.execute()