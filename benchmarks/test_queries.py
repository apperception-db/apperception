import os
import sys
import json
import pandas as pd
import cv2
import psycopg2
import pickle
import os
import shutil
sys.path.append(os.path.join(os.getcwd(),"apperception"))
from new_world import empty_world, World
from camera import Camera
from scenic_generate_df import scenic_generate_df
from camera_config import fetch_camera_config


df_sample_data = pd.DataFrame(columns=["token", "sample_token",	"timestamp", "filename", "is_key_frame", "camera_translation", "camera_rotation", "camera_intrinsic", "ego_rotation", "ego_translation", "scene_name", "frame_order", "heading"])
df_annotation = pd.DataFrame(columns=['token', 'instance_token', 'translation', 'size', 'rotation', 'heading', 'category', 'camera_heading', 'token_sample_data'])
queries = {}


def add_sample(camera_id, cam_x, cam_y, cam_z, cam_heading, object_id, obj_x, obj_y, obj_z, obj_heading):
    df_sample_data.append([camera_id, object_id, 0, "", False, [], [], [], [cam_x, cam_y, cam_z], "test", 0, cam_heading])
    df_annotation.append([object_id, "", "", [obj_x, obj_y, obj_z], [0.621, 0.669, 1.642], [], obj_heading, "vehicle.car"])

def add_query(name, query, expected_result):
    queries[name] = (query, expected_result)


def run_tests():
    #### Deleting the .apperception_cache if it exists, as to avoid DB conflict errors ####
    dirpath = os.path.join('.apperception_cache')
    if os.path.exists(dirpath) and os.path.isdir(dirpath):
        shutil.rmtree(dirpath)
    os.mkdir(dirpath)
    dirpath = os.path.join('output')
    if os.path.exists(dirpath) and os.path.isdir(dirpath):
        shutil.rmtree(dirpath)
    os.mkdir(dirpath)

    World.db.reset()
    name = 'ScenicTestWorld' # world name
    units = 'metrics'      # world units
    user_data_dir = os.path.join("v1.0-mini")

    world = empty_world(name=name)

    scenes = df_sample_data["cameraId"].tolist()
    for scene in scenes:
        config = fetch_camera_config(scene, df_sample_data)
        camera = Camera(config=config, id=scene)
        world = world.add_camera(camera)
        # df_config = df_sample_data[df_sample_data['scene_name'] == scene][['sample_token']]
        # df_ann = df_annotation.join(df_config.set_index('sample_token'), on='sample_token', how='inner')
        world = world.recognize(camera, df_annotation)

    for query_name in queries:
        result = queries[query_name][0](world).get_traj_key()
        expected_result = queries[query_name][1]
        is_pass = True
        for x in expected_result:
            if tuple(x) not in result:
                is_pass = False
        
        if is_pass:
            print("Query \"" + query_name + "\" Passed")
        else:
            print("Query \"" + query_name + "\" FAILED")
            print("\t Result: " + str(result))
            print("\t Expected Result: " + str(expected_result))


add_query("test", lambda world: world, ["nice"])
run_tests()