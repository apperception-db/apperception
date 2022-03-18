import json

import numpy as np
import pandas as pd
from pyquaternion.quaternion import Quaternion


def scenic_generate_df():
    with open("v1.0-mini/v1.0-mini/attribute.json") as f:
        attribute_json = json.load(f)

    with open("v1.0-mini/v1.0-mini/calibrated_sensor.json") as f:
        calibrated_sensor_json = json.load(f)

    with open("v1.0-mini/v1.0-mini/category.json") as f:
        category_json = json.load(f)

    with open("v1.0-mini/v1.0-mini/sample.json") as f:
        sample_json = json.load(f)

    with open("v1.0-mini/v1.0-mini/sample_data.json") as f:
        sample_data_json = json.load(f)

    with open("v1.0-mini/v1.0-mini/sample_annotation.json") as f:
        sample_annotation_json = json.load(f)

    with open("v1.0-mini/v1.0-mini/instance.json") as f:
        instance_json = json.load(f)

    with open("v1.0-mini/v1.0-mini/scene.json") as f:
        scene_json = json.load(f)

    with open("v1.0-mini/v1.0-mini/ego_pose.json") as f:
        ego_pose_json = json.load(f)

    df_sample_data = pd.DataFrame(sample_data_json)
    df_sample_data = df_sample_data[
        [
            "token",
            "sample_token",
            "calibrated_sensor_token",
            "ego_pose_token",
            "timestamp",
            "fileformat",
            "filename",
            "prev",
            "next",
        ]
    ]
    df_sample_data = df_sample_data[df_sample_data["fileformat"] == "jpg"]
    df_sample_data.index = range(len(df_sample_data))

    df_sample = pd.DataFrame(sample_json)
    df_sample.columns = ["sample_token", "timestamp", "prev_sample", "next_sample", "scene_token"]
    df_sample = df_sample[["sample_token", "prev_sample", "next_sample", "scene_token"]]
    df_sample_data = pd.merge(df_sample_data, df_sample, on="sample_token", how="left")

    df_calibrated_sensor = pd.DataFrame(calibrated_sensor_json)
    df_calibrated_sensor.columns = [
        "calibrated_sensor_token",
        "sensor_token",
        "camera_translation",
        "camera_rotation",
        "camera_intrinsic",
    ]
    df_calibrated_sensor = df_calibrated_sensor[
        ["calibrated_sensor_token", "camera_translation", "camera_rotation", "camera_intrinsic"]
    ]
    df_sample_data = pd.merge(
        df_sample_data, df_calibrated_sensor, on="calibrated_sensor_token", how="left"
    )
    df_sample_data = df_sample_data.drop(columns=["calibrated_sensor_token"])

    df_ego_pose = pd.DataFrame(ego_pose_json)
    df_ego_pose.columns = ["ego_pose_token", "timestamp", "ego_rotation", "ego_translation"]
    df_ego_pose = df_ego_pose.drop(columns=["timestamp"])
    df_sample_data = pd.merge(df_sample_data, df_ego_pose, on="ego_pose_token", how="left")
    df_sample_data = df_sample_data.drop(columns=["ego_pose_token"])

    df_scene = pd.DataFrame(scene_json)
    df_scene.columns = [
        "scene_token",
        "log_token",
        "nbr_samples",
        "first_sample_token",
        "last_sample_token",
        "scene_name",
        "description",
    ]
    df_scene = df_scene[["scene_token", "first_sample_token", "last_sample_token", "scene_name"]]
    df_sample_data = pd.merge(df_sample_data, df_scene, on="scene_token", how="left")
    df_sample_data = df_sample_data.drop(columns=["scene_token"])

    df_sample_data["frame_order"] = 0
    for index, row in df_sample_data.iterrows():
        if len(row["prev"]) == 0:
            df_sample_data.loc[index, "frame_order"] = 1
            i = 2
            next_frame_token = row["next"]
            while len(next_frame_token) != 0:
                # print(next_frame_token)
                cur_index = df_sample_data[
                    df_sample_data["token"] == next_frame_token
                ].index.tolist()[0]
                # print(cur_index)
                df_sample_data.loc[cur_index, "frame_order"] = i
                i += 1
                next_frame_token = list(
                    df_sample_data[df_sample_data["token"] == next_frame_token]["next"]
                )[0]

    df_sample_data = df_sample_data.drop(
        columns=[
            "fileformat",
            "prev",
            "next",
            "prev_sample",
            "next_sample",
            "first_sample_token",
            "last_sample_token",
        ]
    )

    df_sample_annotation = pd.DataFrame(sample_annotation_json)
    df_sample_annotation = df_sample_annotation[
        ["token", "sample_token", "instance_token", "translation", "size", "rotation"]
    ]
    heading = []
    for rotation in list(df_sample_annotation["rotation"]):
        heading.append((((Quaternion(rotation).yaw_pitch_roll[0]) * 180 / np.pi) + 360) % 360)
    df_sample_annotation["heading"] = heading

    df_instance = pd.DataFrame(instance_json)
    df_category = pd.DataFrame(category_json)
    df_category.rename(columns={"token": "cat_token"}, inplace=True)
    df_instance = pd.merge(
        df_instance, df_category, left_on="category_token", right_on="cat_token", how="left"
    )
    df_instance = df_instance.drop(
        columns=[
            "category_token",
            "cat_token",
            "nbr_annotations",
            "first_annotation_token",
            "last_annotation_token",
            "description",
        ]
    )
    df_instance.columns = ["instance_token", "category"]
    df_sample_annotation = pd.merge(
        df_sample_annotation, df_instance, on="instance_token", how="left"
    )

    return df_sample_data, df_sample_annotation
