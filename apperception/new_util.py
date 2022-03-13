import ast
import datetime
import os
import time
from typing import Dict, List, Tuple
from tracked_object import TrackedObject
from box import Box
from camera_config import CameraConfig
from pyquaternion import Quaternion

import numpy as np
import lens
import point
import uncompyle6
from video_context import Camera
from video_util import (convert_datetime_to_frame_num, get_video_box,
                        get_video_roi)
from world_executor import (create_transform_matrix,
                            reformat_fetched_world_coords, world_to_pixel)
from scenic_util import bbox_to_data3d, convert_timestamps, join


def create_camera(cam_id, fov):
    # Let's define some attribute for constructing the world first
    name = "traffic_scene"  # world name
    video_file = "./amber_videos/traffic-scene-mini.mp4"  # example video file
    lens_attrs = {"fov": fov, "cam_origin": (0, 0, 0), "skew_factor": 0}
    point_attrs = {
        "p_id": "p1",
        "cam_id": cam_id,
        "x": 0,
        "y": 0,
        "z": 0,
        "time": None,
        "type": "pos",
    }
    camera_attrs = {"ratio": 0.5}
    fps = 30

    fov, res, cam_origin, skew_factor = (
        lens_attrs["fov"],
        [1280, 720],
        lens_attrs["cam_origin"],
        lens_attrs["skew_factor"],
    )

    cam_lens = lens.PinholeLens(res, cam_origin, fov, skew_factor)

    pt_id, cam_id, x, y, z, time, pt_type = (
        point_attrs["p_id"],
        point_attrs["cam_id"],
        point_attrs["x"],
        point_attrs["y"],
        point_attrs["z"],
        point_attrs["time"],
        point_attrs["type"],
    )
    location = point.Point(pt_id, cam_id, (x, y, z), time, pt_type)

    ratio = camera_attrs["ratio"]

    # Ingest the camera to the world
    return Camera(
        cam_id=cam_id,
        point=location,
        ratio=ratio,
        video_file=video_file,
        metadata_id=name + "_" + cam_id,
        lens=cam_lens,
    )


def video_fetch_reformat(fetched_meta):
    result = {}
    for meta in fetched_meta:
        item_id, coordinates, timestamp = meta[0], meta[1:-1], meta[-1]
        if item_id in result:
            result[item_id][0].append(coordinates)
            result[item_id][1].append(timestamp)
        else:
            result[item_id] = [[coordinates], [timestamp]]

    return result


def get_video(metadata_results, cams, start_time, boxed):
    # The cam nodes are raw data from the database
    # TODO: I forget why we used the data from the db instead of directly fetch
    # from the world

    video_files = []
    for cam in cams:
        cam_id, ratio, cam_x, cam_y, cam_z, focal_x, focal_y, fov, skew_factor = (
            cam.cam_id,
            cam.ratio,
            cam.lens.cam_origin[0],
            cam.lens.cam_origin[1],
            cam.lens.cam_origin[2],
            cam.lens.focal_x,
            cam.lens.focal_y,
            cam.lens.fov,
            cam.lens.alpha,
        )
        cam_video_file = cam.video_file
        transform_matrix = create_transform_matrix(focal_x, focal_y, cam_x, cam_y, skew_factor)

        for item_id, vals in metadata_results.items():
            world_coords, timestamps = vals
            # print("timestamps are", timestamps)
            world_coords = reformat_fetched_world_coords(world_coords)
            cam_coords = world_to_pixel(world_coords, transform_matrix)

            vid_times = convert_datetime_to_frame_num(start_time, timestamps)
            # print(vid_times)

            vid_fname = "./output/" + cam.metadata_id + item_id + ".mp4"
            # print(vid_fname)
            if boxed:
                get_video_box(vid_fname, cam_video_file, cam_coords, vid_times)
            else:
                get_video_roi(vid_fname, cam_video_file, cam_coords, vid_times)
            video_files.append(vid_fname)
    print("output video files", ",".join(video_files))
    return video_files


def compile_lambda(pred):
    s = uncompyle6.deparse_code2str(pred.__code__, out=open(os.devnull, "w"))
    tree = ast.parse(s)
    # print(pred.__code__)
    # print(s)
    # apprint(tree)
    subtree = tree.body[0]
    assert isinstance(subtree, ast.Return)

    x_range = []
    y_range = []

    if isinstance(subtree.value, ast.BoolOp):
        left_node = subtree.value.values[0]
        right_node = subtree.value.values[1]

        # parse left
        if isinstance(left_node, ast.Compare):
            cmp_node = left_node
            left = cmp_node.left
            ops = cmp_node.ops
            comparators = cmp_node.comparators

            if (
                len(comparators) == 2
                and isinstance(comparators[0], ast.Attribute)
                and comparators[0].attr == "x"
            ):
                if isinstance(left, ast.BinOp):
                    if isinstance(left.left, ast.Attribute) and left.left.attr == "x":
                        if isinstance(left.op, ast.Sub):
                            assert isinstance(left.right, ast.Num)
                            x_range.append(-left.right.n)
                        elif isinstance(left.op, ast.Add):
                            assert isinstance(left.right, ast.Num)
                            x_range.append(left.right.n)

                if isinstance(comparators[-1], ast.BinOp):
                    right = comparators[-1]
                    if isinstance(right.left, ast.Attribute) and right.left.attr == "x":
                        if isinstance(right.op, ast.Sub):
                            assert isinstance(right.right, ast.Num)
                            x_range.append(-right.right.n)
                        elif isinstance(right.op, ast.Add):
                            assert isinstance(right.right, ast.Num)
                            x_range.append(right.right.n)

        if isinstance(right_node, ast.Compare):
            cmp_node = right_node
            left = cmp_node.left
            ops = cmp_node.ops
            comparators = cmp_node.comparators

            if (
                len(comparators) == 2
                and isinstance(comparators[0], ast.Attribute)
                and comparators[0].attr == "y"
            ):
                if isinstance(left, ast.BinOp):
                    if isinstance(left.left, ast.Attribute) and left.left.attr == "y":
                        if isinstance(left.op, ast.Sub):
                            assert isinstance(left.right, ast.Num)
                            y_range.append(-left.right.n)
                        elif isinstance(left.op, ast.Add):
                            assert isinstance(left.right, ast.Num)
                            y_range.append(left.right.n)

                if isinstance(comparators[-1], ast.BinOp):
                    right = comparators[-1]
                    if isinstance(right.left, ast.Attribute) and right.left.attr == "y":
                        if isinstance(right.op, ast.Sub):
                            assert isinstance(right.right, ast.Num)
                            y_range.append(-right.right.n)
                        elif isinstance(right.op, ast.Add):
                            assert isinstance(right.right, ast.Num)
                            y_range.append(right.right.n)
    return x_range, y_range


def recognize(camera_configs: List[CameraConfig], annotation):
    annotation = annotation.head(500)
    annotations: Dict[str, TrackedObject] = {}
    # sample_token_to_frame_num: Dict[str, str] = {}
    # for config in camera_configs:
    #     if config.frame_id not in sample_token_to_frame_num:
    #         sample_token_to_frame_num[config.frame_id] = []
    #     sample_token_to_frame_num[config.frame_id].append(config.frame_num)
    
    # for a in annotation.itertuples(index=False):
    #     sample_token = a.sample_token
    #     if sample_token not in sample_token_to_frame_num:
    #         continue
    #     frame_nums = sample_token_to_frame_num[sample_token]
    #     item_id = a.instance_token
    #     if item_id not in annotations:
    #         annotations[item_id] = TrackedObject(a.category, [], [])

    #     box = Box(a.translation, a.size, Quaternion(a.rotation))

    #     corners = box.corners()

    #     bbox = np.transpose(corners[:, [3, 7]])
    #     # print(sample_token, item_id)
    #     # print(set(frame_nums))
    #     for frame_num in set(frame_nums):
    #         # TODO: fix this: why are there duplicates
    #         annotations[item_id].bboxes.append(bbox)
    #         annotations[item_id].frame_num.append(int(frame_num))
    #         break
    
    # for item_id in annotations:
    #     frame_num = np.array(annotations[item_id].frame_num)
    #     bboxes = np.array(annotations[item_id].bboxes)

    #     index = frame_num.argsort()

    #     annotations[item_id].frame_num = frame_num[index].tolist()
    #     annotations[item_id].bboxes = bboxes[index, :, :]

    #     print(item_id, len(annotations[item_id].frame_num) == len(set(annotations[item_id].frame_num)))
    for img_file in camera_configs:
        # get bboxes and categories of all the objects appeared in the image file
        sample_token = img_file.frame_id
        frame_num = img_file.frame_num
        all_annotations = annotation[annotation["sample_token"] == sample_token]
        # camera_info = {}
        # camera_info['cameraTranslation'] = img_file['camera_translation']
        # camera_info['cameraRotation'] = img_file['camera_rotation']
        # camera_info['cameraIntrinsic'] = np.array(img_file['camera_intrinsic'])
        # camera_info['egoRotation'] = img_file['ego_rotation']
        # camera_info['egoTranslation'] = img_file['ego_translation']

        for _, ann in all_annotations.iterrows():
            item_id = ann["instance_token"]
            if item_id not in annotations:
                # annotations[item_id] = {"bboxes": [], "frame_num": []}
                # annotations[item_id]["object_type"] = ann["category"]
                annotations[item_id] = TrackedObject(ann["category"], [], [])

            box = Box(ann["translation"], ann["size"], Quaternion(ann["rotation"]))

            corners = box.corners()

            # if item_id == '6dd2cbf4c24b4caeb625035869bca7b5':
            # 	# print("corners", corners)
            # 	# transform_box(box, camera_info)
            # 	# print("transformed box: ", box.corners())
            # 	# corners_2d = box.map_2d(np.array(camera_info['cameraIntrinsic']))
            # 	corners_2d = transformation(box.center, camera_info)
            # 	print("2d_corner: ", corners_2d)
            # 	overlay_bbox("v1.0-mini/samples/CAM_FRONT/n015-2018-07-24-11-22-45+0800__CAM_FRONT__1532402927612460.jpg", corners_2d)

            bbox = [corners[:, 1], corners[:, 7]]
            annotations[item_id].bboxes.append(bbox)
            annotations[item_id].frame_num.append(int(frame_num))

    print("Recognization done, saving to database......")
    return annotations


def add_recognized_objs(
    conn,
    formatted_result: Dict[str, TrackedObject],
    start_time: datetime.datetime,
    camera_id: str
):
    for item_id in formatted_result:
        object_type = formatted_result[item_id].object_type
        recognized_bboxes = np.array(formatted_result[item_id].bboxes)
        tracked_cnt = formatted_result[item_id].frame_num

        top_left = recognized_bboxes[:, 0, :]
        bottom_right = recognized_bboxes[:, 1, :]

        obj_traj = []
        for i in range(len(top_left)):
            current_tl = top_left[i]
            current_br = bottom_right[i]
            obj_traj.append([current_tl.tolist(), current_br.tolist()])

        bboxes_to_postgres(
            conn,
            item_id,
            object_type,
            "default_color",
            start_time,
            tracked_cnt,
            obj_traj,
            camera_id,
        )


def bboxes_to_postgres(
    conn,
    item_id: str,
    object_type: str,
    color: str,
    start_time: datetime.datetime,
    timestamps: List[int],
    bboxes: List[List[List[float]]],
    camera_id: str,
):
    converted_bboxes = [bbox_to_data3d(bbox) for bbox in bboxes]
    pairs = []
    deltas = []
    for meta_box in converted_bboxes:
        pairs.append(meta_box[0])
        deltas.append(meta_box[1:])
    postgres_timestamps = convert_timestamps(start_time, timestamps)
    insert_general_trajectory(
        conn, item_id, object_type, color, postgres_timestamps, bboxes, pairs, camera_id
    )
    # print(f"{item_id} saved successfully")


# Insert general trajectory
def insert_general_trajectory(
    conn,
    item_id: str,
    object_type: str,
    color: str,
    postgres_timestamps: List[str],
    bboxes: List[
        List[List[float]]
    ],  # TODO: should be ((float, float, float), (float, float, float))[]
    pairs: List[Tuple[float, float, float]],
    camera_id: str,
):
    # Creating a cursor object using the cursor() method
    cursor = conn.cursor()

    # Inserting bboxes into Bbox table
    insert_bbox_trajectories_builder = []
    min_tl = np.full(3, np.inf)
    max_br = np.full(3, np.NINF)

    traj_centroids = []

    prevTimestamp = None
    for timestamp, (tl, br), current_point in zip(postgres_timestamps, bboxes, pairs):
        if prevTimestamp == timestamp:
            continue
        prevTimestamp = timestamp
        min_tl = np.minimum(tl, min_tl)
        max_br = np.maximum(br, max_br)

        # Insert bbox
        insert_bbox_trajectories_builder.append(
            f"""
            INSERT INTO General_Bbox (itemId, cameraId, trajBbox)
            VALUES (
                '{item_id}',
                '{camera_id}',
                STBOX 'STBOX ZT(
                    ({join([*tl, timestamp])}),
                    ({join([*br, timestamp])})
                )'
            );
            """
        )

        # Construct trajectory
        traj_centroids.append(f"POINT Z ({join(current_point, ' ')})@{timestamp}")

    # Insert the item_trajectory separately
    insert_trajectory = f"""
    INSERT INTO Item_General_Trajectory (itemId, cameraId, objectType, color, trajCentroids, largestBbox)
    VALUES (
        '{item_id}',
        '{camera_id}',
        '{object_type}',
        '{color}',
        '{{{', '.join(traj_centroids)}}}',
        STBOX 'STBOX Z(
            ({join(min_tl)}),
            ({join(max_br)})
        )'
    );
    """

    cursor.execute(insert_trajectory)
    cursor.execute("".join(insert_bbox_trajectories_builder))

    # Commit your changes in the database
    conn.commit()
