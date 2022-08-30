import os
import pandas as pd
import numpy as np
import math

os.chdir("../")
from apperception.database import database
from apperception.world import empty_world
from apperception.utils import F

### Constants ###
SAMPLING_RATE = 2
CAMERA_ID = "scene-0757"

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
    "cameraTranslationAbs"] #road_direction not included yet

def convert_frame_to_map(frame):
    map_frame = dict(zip(CAMERA_COLUMNS, frames[:12]))
    return map_frame

def transform_to_world(frame_coordinate, ego_translation, ego_rotation):
    ### TODO: get world coordinates
    return frame_coordinate

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
        group.frame_idx = group.frame_idx.add(-1)
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
        for index, row in object_with_ego.iterrows():
            obj_trajectory.append(transform_to_world(frame_coordinate=((row['bbox_left'], row['bbox_top']), 
                                                 (row['bbox_left']+row['bbox_w'], 
                                                  row['bbox_top']+row['bbox_h'])), 
                               ego_translation=row['egoTranslation'],
                               ego_rotation=row['egoRotation']))
        
        obj_info[name]['frame_idx'] = object_with_ego[['frame_idx']]
        obj_info[name]['trajectory'] = obj_trajectory
    return obj_info


def facing_relative(prev_traj_point, next_traj_point, current_ego_heading):
    ### TODO: get direction from adjacent traj points, then calculate the relative degree
    ####### COMPLETE
    diff = next_traj_point - prev_traj_point
    diff_heading = math.degrees(np.arctan2(diff[1], diff[0])) - 90
    result = ((diff_heading - current_ego_heading) % 360 + 360) % 360
    return result

def facing_relative_check(obj_info, threshold):
    for obj_id in obj_info:
        frame_idx = obj_info[obj_id]['frame_idx'].frame_idx
        trajectory = obj_info[obj_id]['trajectory']
        ego_heading = ego_config.iloc[frame_idx.tolist()].egoHeading.tolist()
        print(frame_idx[:len(ego_heading)-1][[facing_relative(trajectory[i], trajectory[i+1], ego_heading[i]) > threshold for i in range(len(ego_heading)-1)]])

"""
Dummy Modules for playing around
"""
class optimizeRoadNetwork:
    def __init__(self, road_network=None):
        self.road_network = road_network
        self.optimized_road_network = self.optimize_road_network()

    def optimize_road_network(self):
        return self.road_network

    def optimize_filter_intersection(self, sample_frames):
        intersection_filtered = []

        # TODO: Connection to DB for each execution might take too much time, do all at same time
        for frame in sampled_frames:
            map_frame = convert_frame_to_map(frame)
            # use sql in order to make use of mobilitydb features. TODO: Find python alternative
            query = f"SELECT TRUE WHERE minDistance('{map_frame['egoTranslation']}', 'intersection') < 10" 
            result = database._execute_query(query)
            if result:
                intersection_filtered.append(frame)

        return intersection_filtered


class optimizeSampling:
    def __init__(self, sampling=None):
        self.sampling = sampling
        self.optimized_sampling = self.optimize_sampling()

    def optimize_sampling(self):
        return self.sampling
    
    def native_sample(self):
        # Sample All Frames from Video at a #
        query = f"SELECT * FROM Cameras WHERE cameraId = '{CAMERA_ID}' ORDER BY frameNum"

        all_frames = database._execute_query(query)
        sampled_frames = all_frames[::SAMPLING_RATE]
        return sampled_frames


class optimizeDecoding:
    def __init__(self, decoding=None):
        self.decoding = decoding
        self.optimized_decoding = self.optimize_decoding()

    def optimize_decoding(self):
        return self.decoding

    def optimize_tracking(self, lst_of_frames):
        # Now the result is written to a txt file, need to fix this later
        from Yolov5_Strong_Detection import sample_frame_tracker
        for frames in lst_of_frames:
            result = sample_frame_tracker.run(frames, save_vid=True)

"""
End to End Optimization Ingestion
Use Case: select cars appears in intersection
                      facing degree d relative to ego
Test Data: only use one video as the test data
"""


class optimizeIngestion:
    def __init__(self):
        self.optimize_road_network = optimizeRoadNetwork()
        self.optimize_sampling = optimizeSampling()
        self.optimize_decoding = optimizeDecoding()

    def run_test(self):
        # 1. Get all frames from video
        all_frames = self.optimize_sampling.native_sample()
        
        
        # 2. Filter out frames that in intersection
        intersection_filtered = self.optimize_road_network.optimize_filter_intersection(all_frames)
        ###TODO:fetch the camera_config corresponding to intersection_filtered 
        ###### COMPLETE
        # query = """SELECT * FROM Cameras 
        #             WHERE filename like 'samples/CAM_FRONT/%2018-08-01-15%' 
        #             ORDER BY frameNum""" 
        # camera_config = database._execute_query(query)
        camera_config_df = pd.DataFrame(intersection_filtered, columns=camera_columns)
        ego_config = camera_config_df[['egoTranslation', 'egoRotation', 'egoHeading']]
        
        
        # 3. Decode filtered_frames and track
        self.optimize_decoding.optimize_tracking([intersection_filtered])
        df = pd.read_csv("../optimization_playground/tracks/CAM_FRONT.txt", sep=" ", header=None, 
                 names=["frame_idx", 
                        "object_id", 
                        "bbox_left", 
                        "bbox_top", 
                        "bbox_w", 
                        "bbox_h", 
                        "None1",
                        "None2",
                        "None3",
                        "None",
                        "object_type"])
        
        obj_info = get_obj_trajectory(df)
        facing_relative_check(obj_info, 0)