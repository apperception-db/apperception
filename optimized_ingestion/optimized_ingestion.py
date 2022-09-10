import os
import pandas as pd
import numpy as np
import math
import cv2

from torchvision import transforms
from pyquaternion import Quaternion

from depth_to_3d import depth_to_3d

# os.chdir("../")
from apperception.database import database
from apperception.world import empty_world
from apperception.utils import F
import sys

if './submodules' not in sys.path:
    sys.path.append('./submodules')

from filters import Filter
from typing import Any, Dict, List, Tuple
from frame import Frame
from pipeline import Pipeline
from monodepth import monodepth

### Constants ###
SAMPLING_RATE = 2
CAMERA_ID = "scene-0757"
TEST_FILE_REG = '%CAM_FRONT/%2018-08-30-15%'
BASE_DIR = '/Users/chanwutk/Documents'
TEST_FILE_DIR = os.path.join(BASE_DIR, 'apperception/data/v1.0-mini/')
TEST_TRACK_FILE = os.path.join(BASE_DIR, "apperception/optimized_ingestion/tracks/CAM_FRONT.txt")
os.remove(TEST_TRACK_FILE) if os.path.exists(TEST_TRACK_FILE) else None

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
    "roadDirection"]

"""
Helper Functions
TODO: Clean Up
"""
def convert_frame_to_map(frames):
    map_frame = dict(zip(CAMERA_COLUMNS, frames[:12]))
    return map_frame

def transform_to_world(frame_coordinates, ego_translation, ego_rotation):
    ### TODO: get world coordinates
    return frame_coordinates

def get_obj_trajectory(tracking_df, ego_config):
    '''
    returned object info is a dictionary that looks like this:
    {object_id:{frame_idx:[], #need to use the frame idx of the video to get the camera config for each frame
                trajectory:[]}
    '''
    obj_info = {}
    grouped_trajectory = tracking_df.groupby(by=["object_id"])
    for name, group in grouped_trajectory:
        obj_info[name] = {}
        
        object_df = group[[
            'frame_idx', 'object_id', 'object_type', 'bbox_left', 'bbox_top', 'bbox_w', 'bbox_h']]
        object_df = object_df.reset_index(drop=True)
        framenums = group.frame_idx.tolist()
        
        ### get ego_config for each framenum
        transformation_config = ego_config.iloc[framenums]
        transformation_config = transformation_config.reset_index(drop=True)
        
        object_with_ego = pd.concat([object_df, transformation_config], axis=1)
        ### for each coordinate, transform
        obj_trajectory = []
        obj_bboxes = []
        obj_3d_camera_trajectory = []
        obj_3d_trajectory = []
        for index, row in object_with_ego.iterrows():
            obj_trajectory.append(transform_to_world(
                                    frame_coordinates=(row['bbox_left']+row['bbox_w']//2, 
                                                       row['bbox_top']+row['bbox_h']//2), 
                                    ego_translation=row['cameraTranslation'],
                                    ego_rotation=row['cameraRotation']))
            obj_bboxes.append(transform_to_world(
                                frame_coordinates=(row['bbox_left'], row['bbox_top'], 
                                                   row['bbox_left']+row['bbox_w'], 
                                                   row['bbox_top']+row['bbox_h']),
                                ego_translation=row['egoTranslation'],
                                ego_rotation=row['egoRotation']))
            x, y, z = row['3d-x'], row['3d-y'], row['3d-z']
            obj_3d_camera_trajectory.append((x, y, z))
            rotated_offset = Quaternion(row['cameraRotation']) \
                .rotate(np.array(x, y, z))
            obj_3d_trajectory.append(np.array(row['cameraTranslation']) + rotated_offset)
        obj_info[name]['frame_idx'] = object_with_ego[['frame_idx']]
        obj_info[name]['trajectory'] = obj_trajectory
        obj_info[name]['bbox'] = obj_bboxes
        obj_info[name]['3d-trajectory'] = obj_3d_camera_trajectory
    return obj_info


def facing_relative(prev_traj_point, next_traj_point, current_ego_heading):
    diff = (next_traj_point[0] - prev_traj_point[0], next_traj_point[1] - prev_traj_point[1])
    diff_heading = math.degrees(np.arctan2(diff[1], diff[0])) - 90
    result = ((diff_heading - current_ego_heading) % 360 + 360) % 360
    return result

def facing_relative_check(obj_info, threshold, ego_config):
    facing_relative_filtered = {}
    for obj_id in obj_info:
        frame_idx = obj_info[obj_id]['frame_idx'].frame_idx
        trajectory = obj_info[obj_id]['trajectory']
        ego_heading = ego_config.iloc[frame_idx.tolist()].egoHeading.tolist()
        filtered_idx = frame_idx[:len(ego_heading)-1][[facing_relative(trajectory[i], trajectory[i+1], ego_heading[i]) > threshold for i in range(len(ego_heading)-1)]]
        facing_relative_filtered[obj_id] = filtered_idx
    return facing_relative_filtered


# Filter used to filter the by how close ego is to an inview segment of some type
class InViewFilter(Filter):
    def __init__(self, distance: float, segment_type: str) -> None:
        self.distance = distance
        self.segment_type = segment_type

    def filter(self, frames: List[Frame], metadata: Dict[Any, Any]) -> Tuple[List[Frame], Dict[Any, Any]]:
        intersection_filtered = []

        # TODO: Connection to DB for each execution might take too much time, do all at same time
        for frame in frames:
            # use sql in order to make use of mobilitydb features. TODO: Find python alternative
            query = f"SELECT TRUE WHERE minDistance('{frame.ego_translation}', '{self.segment_type}') < {self.distance}" 
            result = database._execute_query(query)
            if result:
                intersection_filtered.append(frame)

        return intersection_filtered, metadata


class TrackingFilter(Filter):
    def __init__(self) -> None:
        pass

    def filter(self, frames: List[Frame], metadata: Dict[Any, Any]) -> Tuple[List[Frame], Dict[Any, Any]]:
        import sample_frame_tracker as tracker
        # Now the result is written to a txt file, need to fix this later
        
        camera_config_df = pd.DataFrame([x.get_tuple() for x in frames], columns=CAMERA_COLUMNS)
        ego_config = camera_config_df[['egoTranslation', 'egoRotation', 'egoHeading']]
        camera_config_df.filename = camera_config_df.filename.apply(lambda x: TEST_FILE_DIR + x)
        
        result = tracker.run(camera_config_df.filename.tolist(), save_vid=True)

        md = monodepth()
        depths = [md.eval(frame) for frame in camera_config_df.filename.tolist()]

        df = pd.read_csv(
            TEST_TRACK_FILE,
            sep=",",
            header=None,
            names=[
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
                "object_type"
            ]
        )

        intrinsics = camera_config_df['cameraIntrinsic'].tolist()

        def find_3d_location(row: "pd.Series"):
            x = int(row['bbox_left'] + (row['bbox_w'] / 2))
            y = int(row['bbox_right'] + (row['bbox_h'] / 2))
            idx = row['frame_idx']
            depth = depths[idx][x, y]
            intrinsic = intrinsics[idx]
            return pd.Series(depth_to_3d(x, y, depth, intrinsic), axis=1)

        df[['3d-x', '3d-y', '3d-z']] = df.apply(find_3d_location)
        df.frame_idx = df.frame_idx.add(-1)
        
        obj_info = get_obj_trajectory(df, ego_config)
        facing_relative_filtered = facing_relative_check(obj_info, 0, ego_config)
        for obj_id in facing_relative_filtered:
            ### frame idx of current obj that satisfies the condition
            filtered_idx = facing_relative_filtered[obj_id]
            current_obj_filtered_frames = camera_config_df.filename.iloc[filtered_idx]
            current_obj_bboxes = obj_info[obj_id]['bbox']
            
            # choose codec according to format needed
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
            video = cv2.VideoWriter('../imposed_video/' + str(obj_id) + '_video.avi', fourcc, 1, (1600, 900))

            for i in range(len(current_obj_filtered_frames)):
                img = cv2.imread(current_obj_filtered_frames.iloc[i])
                x, y, x_w, y_h = current_obj_bboxes[filtered_idx.index[i]]
                cv2.rectangle(img,(x,y),(x_w,y_h),(0,255,0),2)
                video.write(img)

            cv2.destroyAllWindows()
            video.release()


if __name__ == "__main__":
    query = f"SELECT * FROM Cameras WHERE filename like '{TEST_FILE_REG}' ORDER BY frameNum"
    all_frames = database._execute_query(query)

    frames = [Frame(x) for x in all_frames]

    pipeline = Pipeline()

    pipeline.add_filter(filter=InViewFilter(distance=10, segment_type="intersection"))
    pipeline.add_filter(filter=TrackingFilter())

    pipeline.run(frames)
