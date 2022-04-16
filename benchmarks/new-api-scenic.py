import os
import sys
import json
import pandas as pd
import cv2
import psycopg2
import pickle
from pypika.dialects import Query, SnowflakeQuery

sys.path.append("./apperception")

from new_world import empty_world, World
from camera import Camera

def main():
    World.db.reset()
    name = 'ScenicWorld' # world name
    units = 'metrics'      # world units
    user_data_dir = os.path.join("v1.0-mini")
    with open('df_sample_data.pickle', "rb") as f:
        df_sample_data = pickle.loads(f.read())
    with open('df_annotation.pickle', "rb") as f:
        df_annotation = pickle.loads(f.read())
    world = empty_world(name=name)

    from camera_config import fetch_camera_config
    # scenes = ["scene-0061", "scene-0103","scene-0553", "scene-0655", "scene-0757", "scene-0796", "scene-0916", "scene-1077", "scene-1094", "scene-1100"]
    scenes = ["scene-0061"]
    for scene in scenes:
        config = fetch_camera_config(scene, df_sample_data)
        camera = Camera(config=config, id=scene)
        world = world.add_camera(camera)
        df_config = df_sample_data[df_sample_data['scene_name'] == scene][['sample_token']]
        df_ann = df_annotation.join(df_config.set_index('sample_token'), on='sample_token', how='inner')
        world = world.recognize(camera, df_ann)

    # car_trajectories = world.filter_traj_type(object_type='vehicle.car').get_traj()
    car_trajectories = world.predicate(lambda obj: obj.object_type == "vehicle.car").get_traj()
    print(car_trajectories)
    car_and_ped_trajs = world.predicate(
            lambda obj: obj.object_type == "vehicle.car" or obj.object_type == "human.pedestrian.adult").get_traj()
    print(car_and_ped_trajs)

    # these two result are the same

def parse():
    '''
    unit test
    '''
    import metadata
    import metadata_util
    import metadata_context
    f = lambda obj: obj.object_type == "vehicle.car"

    pred = metadata_context.Predicate(f)
    pred.new_decompile()

    attribute, operation, comparator, bool_ops, cast_types = pred.get_compile()
    predicate_query = ""
    for i in range(len(attribute)):
        attr = attribute[i]
        op = operation[i]
        comp = comparator[i]
        bool_op = bool_ops[i]
        predicate_query += bool_op + attr + op + comp

def parse_pypika():
    '''
    unit test
    '''
    import metadata
    import metadata_util
    import metadata_context
    f = lambda obj: obj.object_type == "vehicle.car"

    pred = metadata_context.Predicate(f)
    pred.new_decompile()

    attribute, operation, comparator, bool_ops, cast_types = pred.get_compile()

    query = SnowflakeQuery.from_("TMP_TABLE").select("*")
    table, attr = attribute[0].split('.')
    comp = comparator[0]

    return SnowflakeQuery.from_(query).select("*").where(eval(f"query.{attr}=={comp}")) # query.objectType == xxx

if __name__ == "__main__":
    main()
    # parse()
    # print(parse_pypika())
