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


### COLORS FOR OUTPUT ###
GREEN = '\033[92m'
BOLD = '\033[1m'
END = '\033[0m'
RED = '\033[91m'
UNDERLINE = '\033[4m'
BLUE = '\033[94m'

### Dataframes Used To Input Data ###
df_sample_data = pd.DataFrame(columns=["token", "sample_token",	"timestamp", "filename", "is_key_frame", "camera_translation", "camera_rotation", "camera_intrinsic", "ego_rotation", "ego_translation", "scene_name", "frame_order", "heading"])
df_annotation = pd.DataFrame(columns=['sample_token', 'token', 'instance_token', 'translation', 'size', 'rotation', 'heading', 'category', 'camera_heading', 'token_sample_data'])
df_annotation.set_index('sample_token')
queries = {}


def add_sample(df_sample_data, df_annotation, camera_id, cam_x, cam_y, cam_z, cam_heading, object_id, obj_x, obj_y, obj_z, obj_heading, obj_type):
    if not any(df_sample_data.token == camera_id):
        df_sample_data = df_sample_data.append({"token": camera_id, 
                                                "sample_token": camera_id, 
                                                "timestamp": 0, 
                                                "filename": "",
                                                "is_key_frame": False, 
                                                "camera_translation": [0, 0, 0], 
                                                "camera_rotation": [0, 0, 0, 0], 
                                                "camera_intrinsic": [[0, 0, 0], [0, 0, 0], [0, 0, 0]], 
                                                "ego_rotation": [0, 0, 0, 0], 
                                                "ego_translation": [cam_x, cam_y, cam_z], 
                                                "scene_name": camera_id, 
                                                "frame_order": 0, 
                                                "heading": cam_heading}, ignore_index=True)
    
    df_annotation = df_annotation.append({
                                         "sample_token": camera_id,
                                         "token": camera_id, 
                                          "instance_token": object_id, 
                                          "translation": [obj_x, obj_y, obj_z], 
                                          "size": [0.621, 0.669, 1.642], 
                                          "rotation": [0, 0, 0, 0], 
                                          "heading": obj_heading, 
                                          "category": obj_type, 
                                          "camera_heading": cam_heading, 
                                          "token_sample_data": camera_id}, ignore_index=True)
    return df_sample_data, df_annotation

def add_query(name, query, expected_result):
    queries[name] = (query, expected_result)

def run_tests():
    print(BLUE + "\n ----Running Tests----" + END)
    #### Deleting the .apperception_cache if it exists, as to avoid DB conflict errors ####
    dirpath = os.path.join('.apperception_cache')
    if os.path.exists(dirpath) and os.path.isdir(dirpath):
        shutil.rmtree(dirpath)
    os.mkdir(dirpath)
    dirpath = os.path.join('output')
    if os.path.exists(dirpath) and os.path.isdir(dirpath):
        shutil.rmtree(dirpath)
    os.mkdir(dirpath)

    print(BLUE + "\n ----Reseting World----" + END)
    World.db.reset()
    name = 'ScenicTestWorld' # world name
    units = 'metrics'      # world units

    world = empty_world(name=name)

    # with open('df_sample_data.pickle', "rb") as f:
    #     df_sample_data = pickle.loads(f.read())
    # with open('df_annotation.pickle', "rb") as f:
        # df_annotation = pickle.loads(f.read())

    print(BLUE + "\n ----Creating Scenic World----" + END)
    scenes = df_sample_data["scene_name"].tolist()
    for scene in scenes:
        config = fetch_camera_config(scene, df_sample_data)
        camera = Camera(config=config, id=scene)
        world = world.add_camera(camera)
        df_config = df_sample_data[df_sample_data['scene_name'] == scene][['sample_token']]
        df_ann = df_annotation.join(df_config.set_index('sample_token'), on='sample_token', how='inner')
        world = world.recognize(camera, df_ann)

    print(BLUE + "\n ----Getting Camera----" + END)
    cams = world.get_camera()

    print(BLUE + "\n --------Executing Queries--------" + END)
    num_passed = 0
    num_failed = 0
    for query_name in queries:
        result = queries[query_name][0](world).get_traj_key()
        expected_result = queries[query_name][1]
        is_pass = [(x, ) for x in expected_result] == result
        if is_pass:
            print(BOLD + GREEN + "Query \"" + query_name + "\" Passed" + END)
            num_passed += 1
        else:
            print(BOLD + RED + "Query \"" + query_name + "\" FAILED:" + END)
            print("\t Result: " + str(result))
            print("\t Expected Result: " + str([(x, ) for x in expected_result]))
            num_failed += 1
    
    print(BLUE + "\n ----Queries Executed----" + END)
    print("\t" + GREEN + "Passed: " + str(num_passed) + END)
    print("\t" + RED + "Failed: " + str(num_failed) + END)



def query1(world):
    filtered_world = world.filter_traj_type("vehicle.car")

    filtered_world = filtered_world.filter_pred_relative_to_type(pred=lambda obj: (cam.x - 0) <= obj.x <= (cam.x + 11) 
                                                                     and (cam.y - 0) <= obj.y <= (cam.y + 11))
    return filtered_world



########################## FILTER_RELATIVE_TO_TYPE ##########################
df_sample_data, df_annotation = add_sample(df_sample_data, df_annotation, "camera1", 0, 0, 0, 0, "Object At (10, 10)", 10, 10, 0, 0, "Camera 0 Heading")
df_sample_data, df_annotation = add_sample(df_sample_data, df_annotation, "camera1", 0, 0, 0, 0, "Object At (10, -10)", 10, -10, 0, 0, "Camera 0 Heading")
df_sample_data, df_annotation = add_sample(df_sample_data, df_annotation, "camera1", 0, 0, 0, 0, "Object At (-10, -10)", -10, -10, 0, 0, "Camera 0 Heading")
df_sample_data, df_annotation = add_sample(df_sample_data, df_annotation, "camera1", 0, 0, 0, 0, "Object At (-10, 10)", -10, 10, 0, 0, "Camera 0 Heading")

add_query("Camera 0 Heading: Object 10 Above, 10 Right", lambda world: world.filter_traj_type("Camera 0 Heading")
                                                         .filter_pred_relative_to_type(pred=lambda obj: (cam.x - 0) <= obj.x <= (cam.x + 11) 
                                                                                                   and (cam.y - 0) <= obj.y <= (cam.y + 11)), 
         ["Object At (10, 10)"])
add_query("Camera 0 Heading: Object 10 Below, 10 Right", lambda world: world.filter_traj_type("Camera 0 Heading")
                                                         .filter_pred_relative_to_type(pred=lambda obj: (cam.x - 0) <= obj.x <= (cam.x + 11) 
                                                                                                   and (cam.y - 11) <= obj.y <= (cam.y + 0)),  
         ["Object At (10, -10)"])
add_query("Camera 0 Heading: Object 10 Below, 10 Left", lambda world: world.filter_traj_type("Camera 0 Heading")
                                                         .filter_pred_relative_to_type(pred=lambda obj: (cam.x - 11) <= obj.x <= (cam.x + 0) 
                                                                                                   and (cam.y - 11) <= obj.y <= (cam.y + 0)), 
         ["Object At (-10, -10)"])
add_query("Camera 0 Heading: Object 10 Above, 10 Left", lambda world: world.filter_traj_type("Camera 0 Heading")
                                                         .filter_pred_relative_to_type(pred=lambda obj: (cam.x - 11) <= obj.x <= (cam.x + 0) 
                                                                                                   and (cam.y - 0) <= obj.y <= (cam.y + 11)), 
         ["Object At (-10, 10)"])


df_sample_data, df_annotation = add_sample(df_sample_data, df_annotation, "camera2", 50, 50, 0, 90, "Object At (50, 40)", 50, 40, 0, 0, "Camera 90 Heading")
df_sample_data, df_annotation = add_sample(df_sample_data, df_annotation, "camera2", 50, 50, 0, 90, "Object At (50, 60)", 50, 60, 0, 0, "Camera 90 Heading")
df_sample_data, df_annotation = add_sample(df_sample_data, df_annotation, "camera2", 50, 50, 0, 90, "Object At (40, 50)", 40, 50, 0, 0, "Camera 90 Heading")
df_sample_data, df_annotation = add_sample(df_sample_data, df_annotation, "camera2", 50, 50, 0, 90, "Object At (60, 50)", 60, 50, 0, 0, "Camera 90 Heading")

add_query("Camera 90 Heading: Object 10 Above", lambda world: world.filter_traj_type("Camera 90 Heading")
                                                         .filter_pred_relative_to_type(pred=lambda obj: (cam.x - 1) <= obj.x <= (cam.x + 1) 
                                                                                                   and (cam.y - 0) <= obj.y <= (cam.y + 11)), 
         ["Object At (60, 50)"])
add_query("Camera 90 Heading: Object 10 Below", lambda world: world.filter_traj_type("Camera 90 Heading")
                                                         .filter_pred_relative_to_type(pred=lambda obj: (cam.x - 1) <= obj.x <= (cam.x + 1) 
                                                                                                   and (cam.y - 11) <= obj.y <= (cam.y + 0)), 
         ["Object At (40, 50)"])
add_query("Camera 90 Heading: Object 10 Left", lambda world: world.filter_traj_type("Camera 90 Heading")
                                                         .filter_pred_relative_to_type(pred=lambda obj: (cam.x - 11) <= obj.x <= (cam.x + 0) 
                                                                                                   and (cam.y - 1) <= obj.y <= (cam.y + 1)), 
         ["Object At (50, 60)"])
add_query("Camera 90 Heading: Object 10 Right", lambda world: world.filter_traj_type("Camera 90 Heading")
                                                         .filter_pred_relative_to_type(pred=lambda obj: (cam.x - 0) <= obj.x <= (cam.x + 11) 
                                                                                                   and (cam.y - 1) <= obj.y <= (cam.y + 1)), 
         ["Object At (50, 40)"])

df_sample_data, df_annotation = add_sample(df_sample_data, df_annotation, "camera3", -50, -50, 0, 180, "Object At (-50, -40)", -50, -40, 0, 0, "Camera 180 Heading")
df_sample_data, df_annotation = add_sample(df_sample_data, df_annotation, "camera3", -50, -50, 0, 180, "Object At (-50, -60)", -50, -60, 0, 0, "Camera 180 Heading")
df_sample_data, df_annotation = add_sample(df_sample_data, df_annotation, "camera3", -50, -50, 0, 180, "Object At (-40, -50)", -40, -50, 0, 0, "Camera 180 Heading")
df_sample_data, df_annotation = add_sample(df_sample_data, df_annotation, "camera3", -50, -50, 0, 180, "Object At (-60, -50)", -60, -50, 0, 0, "Camera 180 Heading")

add_query("Camera 180 Heading: Object 10 Above", lambda world: world.filter_traj_type("Camera 180 Heading")
                                                         .filter_pred_relative_to_type(pred=lambda obj: (cam.x - 1) <= obj.x <= (cam.x + 1) 
                                                                                                   and (cam.y - 0) <= obj.y <= (cam.y + 11)), 
         ["Object At (-50, -60)"])
add_query("Camera 180 Heading: Object 10 Below", lambda world: world.filter_traj_type("Camera 180 Heading")
                                                         .filter_pred_relative_to_type(pred=lambda obj: (cam.x - 1) <= obj.x <= (cam.x + 1) 
                                                                                                   and (cam.y - 11) <= obj.y <= (cam.y + 0)), 
         ["Object At (-50, -40)"])
add_query("Camera 180 Heading: Object 10 Left", lambda world: world.filter_traj_type("Camera 180 Heading")
                                                         .filter_pred_relative_to_type(pred=lambda obj: (cam.x - 11) <= obj.x <= (cam.x + 0) 
                                                                                                   and (cam.y - 1) <= obj.y <= (cam.y + 1)), 
         ["Object At (-40, -50)"])
add_query("Camera 180 Heading: Object 10 Right", lambda world: world.filter_traj_type("Camera 180 Heading")
                                                         .filter_pred_relative_to_type(pred=lambda obj: (cam.x - 0) <= obj.x <= (cam.x + 11) 
                                                                                                   and (cam.y - 1) <= obj.y <= (cam.y + 1)), 
         ["Object At (-60, -50)"])

df_sample_data, df_annotation = add_sample(df_sample_data, df_annotation, "camera4", 400, 400, 0, 270, "Object At (400, 390)", 400, 390, 0, 0, "Camera 270 Heading")
df_sample_data, df_annotation = add_sample(df_sample_data, df_annotation, "camera4", 400, 400, 0, 270, "Object At (400, 410)", 400, 410, 0, 0, "Camera 270 Heading")
df_sample_data, df_annotation = add_sample(df_sample_data, df_annotation, "camera4", 400, 400, 0, 270, "Object At (390, 400)", 390, 400, 0, 0, "Camera 270 Heading")
df_sample_data, df_annotation = add_sample(df_sample_data, df_annotation, "camera4", 400, 400, 0, 270, "Object At (410, 400)", 410, 400, 0, 0, "Camera 270 Heading")

add_query("Camera 270 Heading: Object 10 Above", lambda world: world.filter_traj_type("Camera 270 Heading")
                                                         .filter_pred_relative_to_type(pred=lambda obj: (cam.x - 1) <= obj.x <= (cam.x + 1) 
                                                                                                   and (cam.y - 0) <= obj.y <= (cam.y + 11)), 
         ["Object At (390, 400)"])
add_query("Camera 270 Heading: Object 10 Below", lambda world: world.filter_traj_type("Camera 270 Heading")
                                                         .filter_pred_relative_to_type(pred=lambda obj: (cam.x - 1) <= obj.x <= (cam.x + 1) 
                                                                                                   and (cam.y - 11) <= obj.y <= (cam.y + 0)), 
         ["Object At (410, 400)"])
add_query("Camera 270 Heading: Object 10 Left", lambda world: world.filter_traj_type("Camera 270 Heading")
                                                         .filter_pred_relative_to_type(pred=lambda obj: (cam.x - 11) <= obj.x <= (cam.x + 0) 
                                                                                                   and (cam.y - 1) <= obj.y <= (cam.y + 1)), 
         ["Object At (400, 390)"])
add_query("Camera 270 Heading: Object 10 Right", lambda world: world.filter_traj_type("Camera 270 Heading")
                                                         .filter_pred_relative_to_type(pred=lambda obj: (cam.x - 0) <= obj.x <= (cam.x + 11) 
                                                                                                   and (cam.y - 1) <= obj.y <= (cam.y + 1)), 
         ["Object At (400, 410)"])
########################## ####################### ##########################


run_tests()