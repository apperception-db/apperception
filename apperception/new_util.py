import ast
import os
from datetime import datetime
from types import FunctionType
from typing import Dict, List, Tuple

import lens
import numpy as np
import point
import uncompyle6
from box import Box
from camera_config import CameraConfig
from pypika.dialects import SnowflakeQuery
from pyquaternion import Quaternion
from scenic_util import bbox_to_data3d, join
from tracked_object import TrackedObject
from video_context import Camera
from video_util import (convert_datetime_to_frame_num, get_video_box,
                        get_video_roi)
from world_executor import (create_transform_matrix,
                            reformat_fetched_world_coords, world_to_pixel)


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
    # fps = 30

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
        cam_x, cam_y, focal_x, focal_y, skew_factor = (
            cam.lens.cam_origin[0],
            cam.lens.cam_origin[1],
            cam.lens.focal_x,
            cam.lens.focal_y,
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
            # ops = cmp_node.ops
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
            # ops = cmp_node.ops
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
    annotations: Dict[str, TrackedObject] = {}
    sample_token_to_time: Dict[str, int] = {}
    for config in camera_configs:
        if config.frame_id in sample_token_to_time:
            raise Exception("duplicate frame_id")
        sample_token_to_time[config.frame_id] = config.timestamp

    for a in annotation.itertuples(index=False):
        sample_data_token = a.token_sample_data
        if sample_data_token not in sample_token_to_time:
            continue
        timestamp = sample_token_to_time[sample_data_token]
        item_id = a.instance_token
        if item_id not in annotations:
            annotations[item_id] = TrackedObject(a.category, [], [])

        box = Box(a.translation, a.size, Quaternion(a.rotation))

        corners = box.corners()
        bbox = np.transpose(corners[:, [3, 7]])

        annotations[item_id].bboxes.append(bbox)
        annotations[item_id].timestamps.append(timestamp)
        annotations[item_id].itemHeading.append(a.heading)

    for item_id in annotations:
        timestamps = np.array(annotations[item_id].timestamps)
        bboxes = np.array(annotations[item_id].bboxes)
        itemHeadings = np.array(annotations[item_id].itemHeading)

        index = timestamps.argsort()

        annotations[item_id].timestamps = timestamps[index].tolist()
        annotations[item_id].bboxes = [bboxes[i, :, :] for i in index]
        annotations[item_id].itemHeading = itemHeadings[index].tolist()

    print("Recognization done, saving to database......")
    return annotations


def add_recognized_objs(conn, formatted_result: Dict[str, TrackedObject], camera_id: str):
    for item_id in formatted_result:
        object_type = formatted_result[item_id].object_type
        recognized_bboxes = np.array(formatted_result[item_id].bboxes)
        timestamps = formatted_result[item_id].timestamps
        itemHeading_list = formatted_result[item_id].itemHeading

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
            timestamps,
            obj_traj,
            camera_id,
            itemHeading_list,
        )


def bboxes_to_postgres(
    conn,
    item_id: str,
    object_type: str,
    color: str,
    timestamps: List[int],
    bboxes: List[List[List[float]]],
    camera_id: str,
    itemHeading_list: List[int],
):
    converted_bboxes = [bbox_to_data3d(bbox) for bbox in bboxes]
    pairs = []
    deltas = []
    for meta_box in converted_bboxes:
        pairs.append(meta_box[0])
        deltas.append(meta_box[1:])
    insert_general_trajectory(
        conn,
        item_id,
        object_type,
        color,
        [str(datetime.fromtimestamp(t / 1000000.0)) for t in timestamps],
        bboxes,
        pairs,
        camera_id,
        itemHeading_list,
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
    itemHeading_list: List[int],
):

    # Creating a cursor object using the cursor() method
    cursor = conn.cursor()

    # Inserting bboxes into Bbox table
    insert_bbox_trajectories_builder = []
    min_tl = np.full(3, np.inf)
    max_br = np.full(3, np.NINF)

    traj_centroids = []
    itemHeadings = []
    prevTimestamp = None
    for timestamp, (tl, br), current_point, curItemHeading in zip(
        postgres_timestamps, bboxes, pairs, itemHeading_list
    ):
        if prevTimestamp == timestamp:
            continue
        prevTimestamp = timestamp
        min_tl = np.minimum(tl, min_tl)
        max_br = np.maximum(br, max_br)

        # Insert bbox
        insert_bbox_trajectories_builder.append(
            f"""(
            '{item_id}',
            '{camera_id}',
            STBOX 'STBOX ZT(
                ({join([*tl, timestamp])}),
                ({join([*br, timestamp])})
            )'
        )"""
        )

        # Construct trajectory
        traj_centroids.append(f"POINT Z ({join(current_point, ' ')})@{timestamp}")
        itemHeadings.append(f"{curItemHeading}@{timestamp}")

    # Insert the item_trajectory separately
    insert_trajectory = f"""
    INSERT INTO Item_General_Trajectory (itemId, cameraId, objectType, color, trajCentroids, largestBbox, itemHeadings)
    VALUES (
        '{item_id}',
        '{camera_id}',
        '{object_type}',
        '{color}',
        tgeompoint '{{[{', '.join(traj_centroids)}]}}',
        STBOX 'STBOX Z(
            ({join(min_tl)}),
            ({join(max_br)})
        )',
        tfloat '{{[{', '.join(itemHeadings)}]}}'
    );
    """

    cursor.execute(insert_trajectory)
    if len(insert_bbox_trajectories_builder):
        cursor.execute(
            f"""
        INSERT INTO General_Bbox (itemId, cameraId, trajBbox)
        VALUES {",".join(insert_bbox_trajectories_builder)}
        """
        )
        # cursor.execute(",".join(insert_bbox_trajectories_builder))

    # Commit your changes in the database
    conn.commit()


def parse_predicate(query: SnowflakeQuery, f: FunctionType):
    import metadata_context

    pred = metadata_context.Predicate(f)
    pred.new_decompile()

    attribute, operation, comparator, bool_ops, cast_types = pred.get_compile()

    if len(bool_ops) == 0:
        table, attr = attribute[0].split(".")
        comp = comparator[0]

        return f"query.{attr}=={comp}"  # query.objectType == xxx

    else:
        assert len(bool_ops) + 1 == len(attribute)
        # import pdb; pdb.set_trace()
        table, attr = attribute[0].split(".")
        comp = comparator[0]

        q_str = f"(query.{attr}=={comp})"
        for i in range(len(bool_ops)):
            q_str += " " + bool_ops[i] + " "
            _, attr = attribute[i + 1].split(".")
            comp = comparator[i + 1]
            q_str += f"(query.{attr} == {comp})"

        return q_str
