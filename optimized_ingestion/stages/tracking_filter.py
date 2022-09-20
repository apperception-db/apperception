import math
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from bitarray import bitarray
from pyquaternion import Quaternion

from ..camera_config import CameraConfig
from ..monodepth import monodepth
from ..trackers import yolov5_strongsort_osnet_tracker as tracker
from ..utils.depth_to_3d import depth_to_3d
from .stage import Stage
if TYPE_CHECKING:
    from ..payload import Payload

BASE_DIR = Path(__file__).resolve().parent.parent
TEST_FILE_DIR = os.path.join(BASE_DIR, "data/v1.0-mini/")


CAMERA_COLUMNS = [
    "cameraId",
    "frameId",
    "frameNum",
    "filename",
    "cameraTranslation",
    "cameraRotation",
    "cameraIntrinsic",
    "egoTranslation",
    "egoRotation",
    "timestamp",
    "cameraHeading",
    "egoHeading",
    "roadDirection",
]

"""
Helper Functions
TODO: Clean Up
"""


def convert_frame_to_map(frames):
    map_frame = dict(zip(CAMERA_COLUMNS, frames[:12]))
    return map_frame


def transform_to_world(frame_coordinates, ego_translation, ego_rotation):
    # TODO: get world coordinates
    return frame_coordinates


def get_obj_trajectory(tracking_df, ego_config):
    """
    returned object info is a dictionary that looks like this:
    {object_id:{frame_idx:[], #need to use the frame idx of the video to get the camera config for each frame
                trajectory:[]}
    """
    obj_info = {}
    grouped_trajectory = tracking_df.groupby(by=["object_id"])
    for name, group in grouped_trajectory:
        obj_info[name] = {}

        object_df = group[
            [
                "frame_idx",
                "object_id",
                "object_type",
                "bbox_left",
                "bbox_top",
                "bbox_w",
                "bbox_h",
                "3d-x",
                "3d-y",
                "3d-z",
            ]
        ]
        object_df = object_df.reset_index(drop=True)
        framenums = group.frame_idx.tolist()

        # get ego_config for each framenum
        transformation_config = ego_config.iloc[framenums]
        transformation_config = transformation_config.reset_index(drop=True)

        object_with_ego = pd.concat([object_df, transformation_config], axis=1)
        # for each coordinate, transform
        obj_trajectory = []
        obj_bboxes = []
        obj_3d_camera_trajectory = []
        obj_3d_trajectory = []
        for index, row in object_with_ego.iterrows():
            obj_trajectory.append(
                transform_to_world(
                    frame_coordinates=(
                        row["bbox_left"] + row["bbox_w"] // 2,
                        row["bbox_top"] + row["bbox_h"] // 2,
                    ),
                    ego_translation=row["cameraTranslation"],
                    ego_rotation=row["cameraRotation"],
                )
            )
            obj_bboxes.append(
                transform_to_world(
                    frame_coordinates=(
                        row["bbox_left"],
                        row["bbox_top"],
                        row["bbox_left"] + row["bbox_w"],
                        row["bbox_top"] + row["bbox_h"],
                    ),
                    ego_translation=row["egoTranslation"],
                    ego_rotation=row["egoRotation"],
                )
            )
            x, y, z = row["3d-x"], row["3d-y"], row["3d-z"]
            obj_3d_camera_trajectory.append((x, y, z))
            rotated_offset = Quaternion(row["cameraRotation"]).rotate(np.array([x, y, z]))
            obj_3d_trajectory.append(np.array(row["cameraTranslation"]) + rotated_offset)
        obj_info[name]["frame_idx"] = object_with_ego[["frame_idx"]]
        obj_info[name]["trajectory"] = obj_trajectory
        obj_info[name]["bbox"] = obj_bboxes
        obj_info[name]["3d-trajectory"] = obj_3d_camera_trajectory
    return obj_info


def facing_relative(prev_traj_point, next_traj_point, current_ego_heading):
    diff = (next_traj_point[0] - prev_traj_point[0], next_traj_point[1] - prev_traj_point[1])
    diff_heading = math.degrees(np.arctan2(diff[1], diff[0])) - 90
    result = ((diff_heading - current_ego_heading) % 360 + 360) % 360
    if result > 180:
        result -= 360
    return result


def facing_relative_check(obj_info, threshold, ego_config):
    facing_relative_filtered = {}
    for obj_id in obj_info:
        frame_idx = obj_info[obj_id]["frame_idx"].frame_idx
        trajectory = obj_info[obj_id]["trajectory"]
        ego_heading = ego_config.iloc[frame_idx.tolist()].egoHeading.tolist()
        filtered_idx = frame_idx[: len(ego_heading) - 1][
            [
                facing_relative(trajectory[i], trajectory[i + 1], ego_heading[i]) > threshold
                for i in range(len(ego_heading) - 1)
            ]
        ]
        facing_relative_filtered[obj_id] = filtered_idx
    return facing_relative_filtered


RESULT_COLUMNS = [
    "frame_idx",
    "object_id",
    "bbox_left",
    "bbox_top",
    "bbox_w",
    "bbox_h",
    "None1",
    "None2",
    "None3",
    "None",
    "object_type",
    "conf",
]


class TrackingFilter(Stage):
    def __init__(self) -> None:
        pass

    def __call__(self, payload: "Payload") -> "Tuple[Optional[bitarray], Optional[list]]":
        md = monodepth()
        depths = []
        video = cv2.VideoCapture(payload.video.videofile)
        idx = 0
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break
            if payload.keep[idx]:
                depths.append(md.eval(frame))
            else:
                depths.append(None)
            idx += 1
        assert idx == len(payload.video)
        video.release()
        cv2.destroyAllWindows()
        # with open("./depths.pickle", "wb") as pwrite:
        #     pickle.dump(depths, pwrite)
        # with open("./depths.pickle", "rb") as pread:
        #     depths = pickle.load(pread)

        results = tracker.track(payload)
        # with open("./results.pickle", "wb") as pwrite:
        #     pickle.dump(results, pwrite)
        # with open("./results.pickle", "rb") as pread:
        #     results = pickle.load(pread)

        _results = [{k: v for k, v in zip(RESULT_COLUMNS, r)} for r in results]
        _results = sorted(_results, key=lambda r: r["frame_idx"])
        metadata: "List[Any]" = [None for _ in range(len(depths))]
        trajectories: "Dict[Any, Any]" = {}

        for row in _results:
            x = int(row["bbox_left"] + (row["bbox_w"] / 2))
            y = int(row["bbox_top"] + (row["bbox_h"] / 2))
            idx = int(row["frame_idx"])
            depth = depths[idx][y, x]
            camera = payload.video[idx]
            intrinsic = camera.camera_intrinsic
            row["3d-to-camera"] = depth_to_3d(x, y, depth, intrinsic)
            rotated_offset = Quaternion(camera.camera_rotation).rotate(
                np.array(row["3d-to-camera"])
            )
            row["3d"] = np.array(camera.camera_translation) + rotated_offset

            if metadata[idx] is None:
                metadata[idx] = {}
            if "trackings" not in metadata[idx]:
                metadata[idx]["trackings"] = []
            metadata[idx]["trackings"].append(row)

            if row["object_id"] not in trajectories:
                trajectories[row["object_id"]] = []
            trajectories[row["object_id"]].append(row)

        for trajectory in trajectories.values():
            last = len(trajectory) - 1
            for i, t in enumerate(trajectory):
                if i > 0:
                    t["prev"] = trajectory[i - 1]
                if i < last:
                    t["next"] = trajectory[i + 1]

        keep = bitarray(payload.keep)
        for i, (f, m) in enumerate(zip(payload.video, metadata)):
            if m is None:
                continue
            trackings = m["trackings"]
            for tracking in trackings:
                if "prev" in tracking:
                    _from = tracking["prev"]
                else:
                    _from = tracking

                if "next" in tracking:
                    _to = tracking["next"]
                else:
                    _to = tracking

                angle = facing_relative(_from["3d"], _to["3d"], f.ego_heading)
                if (50 < angle and angle < 135) or (-135 < angle and angle < -50):
                    tracking["matched"] = True
                else:
                    tracking["matched"] = False
                    keep[i] = 0

        return keep, metadata

    def filter(
        self, frames: List[CameraConfig], metadata: Dict[Any, Any]
    ) -> Tuple[List[CameraConfig], Dict[Any, Any]]:
        import trackers.yolov5_strongsort_osnet_tracker as tracker

        # Now the result is written to a txt file, need to fix this later

        camera_config_df = pd.DataFrame([tuple(x) for x in frames], columns=CAMERA_COLUMNS)
        ego_config = camera_config_df[
            ["egoTranslation", "egoRotation", "egoHeading", "cameraTranslation", "cameraRotation"]
        ]
        camera_config_df.filename = camera_config_df.filename.apply(lambda x: TEST_FILE_DIR + x)

        md = monodepth()
        depths = [md.eval(frame) for frame in camera_config_df.filename.tolist()]
        # with open("./depths.pickle", "wb") as pwrite:
        #     pickle.dump(depths, pwrite)
        # with open("./depths.pickle", "rb") as pread:
        #     depths = pickle.load(pread)

        results = tracker.track(camera_config_df.filename.tolist())
        # with open("./results.pickle", "wb") as pwrite:
        #     pickle.dump(results, pwrite)
        # with open("./results.pickle", "rb") as pread:
        #     results = pickle.load(pread)

        df = pd.DataFrame(
            results,
            columns=[
                "frame_idx",
                "object_id",
                "bbox_left",
                "bbox_top",
                "bbox_w",
                "bbox_h",
                "None1",
                "None2",
                "None3",
                "None",
                "object_type",
                "conf",
            ],
        )

        intrinsics = camera_config_df["cameraIntrinsic"].tolist()

        def find_3d_location(row: "pd.Series"):
            x = int(row["bbox_left"] + (row["bbox_w"] / 2))
            y = int(row["bbox_top"] + (row["bbox_h"] / 2))
            idx = row["frame_idx"]
            depth = depths[idx][y, x]
            intrinsic = intrinsics[idx]
            return pd.Series(depth_to_3d(x, y, depth, intrinsic))

        df[["3d-x", "3d-y", "3d-z"]] = df.apply(find_3d_location, axis=1)
        df.frame_idx = df.frame_idx.add(-1)

        obj_info = get_obj_trajectory(df, ego_config)
        facing_relative_filtered = facing_relative_check(obj_info, 0, ego_config)
        for obj_id in facing_relative_filtered:
            # frame idx of current obj that satisfies the condition
            filtered_idx = facing_relative_filtered[obj_id]
            current_obj_filtered_frames = camera_config_df.filename.iloc[filtered_idx]
            current_obj_bboxes = obj_info[obj_id]["bbox"]

            # choose codec according to format needed
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video = cv2.VideoWriter(
                "../imposed_video/" + str(obj_id) + "_video.avi", fourcc, 1, (1600, 900)
            )

            for i in range(len(current_obj_filtered_frames)):
                img = cv2.imread(current_obj_filtered_frames.iloc[i])
                x, y, x_w, y_h = current_obj_bboxes[filtered_idx.index[i]]
                cv2.rectangle(img, (int(x), int(y)), (int(x_w), int(y_h)), (0, 255, 0), 2)
                video.write(img)

            cv2.destroyAllWindows()
            video.release()
        return frames, metadata
