import lens
import point
from video_context import Camera
from video_util import (convert_datetime_to_frame_num, get_video_box,
                        get_video_roi)
from world_executor import (create_transform_matrix,
                            reformat_fetched_world_coords, world_to_pixel)
import uncompyle6
import ast
import os

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
