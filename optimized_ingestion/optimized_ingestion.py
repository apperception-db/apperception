import os
import pandas as pd
import numpy as np
import math

os.chdir("../")
from apperception.database import database
from apperception.world import empty_world
from apperception.utils import F
import sys
sys.path.insert(1, './Yolov5_StrongSORT_OSNet')

### Constants ###
SAMPLING_RATE = 2
CAMERA_ID = "scene-0757"
TEST_FILE_REG = 'samples/CAM_FRONT/%2018-08-01-15%'
TEST_FILE_DIR = '/home/yongming/workspace/research/apperception/v1.0-mini/'
TEST_TRACK_FILE = "/home/yongming/workspace/research/apperception_new_local/apperception/optimized_ingestion/tracks/CAM_FRONT.txt"
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
    "cameraTranslationAbs",
    "roadDirection"]

"""
Helper Functions
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
        for index, row in object_with_ego.iterrows():
            obj_trajectory.append(transform_to_world(
                                    frame_coordinates=(row['bbox_left']+row['bbox_w']//2, 
                                                       row['bbox_top']+row['bbox_h']//2), 
                                    ego_translation=row['egoTranslation'],
                                    ego_rotation=row['egoRotation']))
            obj_bboxes.append(transform_to_world(
                                frame_coordinates=(row['bbox_left'], row['bbox_top'], 
                                                   row['bbox_left']+row['bbox_w'], 
                                                   row['bbox_top']+row['bbox_h']),
                                ego_translation=row['egoTranslation'],
                                ego_rotation=row['egoRotation']))
        obj_info[name]['frame_idx'] = object_with_ego[['frame_idx']]
        obj_info[name]['trajectory'] = obj_trajectory
        obj_info[name]['bbox'] = obj_bboxes
    return obj_info


def facing_relative(prev_traj_point, next_traj_point, current_ego_heading):
    ### TODO: get direction from adjacent traj points, then calculate the relative degree
    ####### COMPLETE
    
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

"""
Dummy Modules for playing around
"""

class optimizeSampling:
    def __init__(self, sampling=None):
        self.sampling = sampling
        self.optimized_sampling = self.optimize_sampling()

    def optimize_sampling(self):
        return self.sampling
    
    def naive_sample(self):
        print("start sampling")
        # Sample All Frames from Video at a #
        query = f"SELECT * FROM Cameras WHERE filename like '{TEST_FILE_REG}' ORDER BY frameNum"

        all_frames = database._execute_query(query)
        print("length of all frames", len(all_frames))
        sampled_frames = all_frames[::SAMPLING_RATE]
        print("length of sampled_frames", len(sampled_frames))
        return sampled_frames

class optimizeRoadNetwork:
    def __init__(self, road_network=None):
        self.road_network = road_network
        self.optimized_road_network = self.optimize_road_network()

    def optimize_road_network(self):
        return self.road_network

    def optimize_filter_intersection(self, sample_frames):
        intersection_filtered = []

        # TODO: Connection to DB for each execution might take too much time, do all at same time
        cnt = 0
        for frame in sample_frames:
            print(cnt)
            cnt += 1
            map_frame = convert_frame_to_map(frame)
            # use sql in order to make use of mobilitydb features. TODO: Find python alternative
            query = f"SELECT TRUE WHERE minDistance('{map_frame['egoTranslation']}', 'intersection') < 10" 
            result = database._execute_query(query)
            if result:
                intersection_filtered.append(frame)

        return intersection_filtered

class optimizeDecoding:
    def __init__(self, decoding=None):
        self.decoding = decoding
        self.optimized_decoding = self.optimize_decoding()

    def optimize_decoding(self):
        return self.decoding

    def optimize_tracking(self, lst_of_frames):
        import sample_frame_tracker
        # Now the result is written to a txt file, need to fix this later
        for frames in lst_of_frames:
            print("frames are", frames)
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
        all_frames = self.optimize_sampling.naive_sample()
        
        
        # 2. Filter out frames that in intersection
        intersection_filtered = self.optimize_road_network.optimize_filter_intersection(all_frames)
        ###TODO:fetch the camera_config corresponding to intersection_filtered 
        ###### COMPLETE
        # query = """SELECT * FROM Cameras 
        #             WHERE filename like 'samples/CAM_FRONT/%2018-08-01-15%' 
        #             ORDER BY frameNum""" 
        # camera_config = database._execute_query(query)
        camera_config_df = pd.DataFrame(intersection_filtered, columns=CAMERA_COLUMNS)
        ego_config = camera_config_df[['egoTranslation', 'egoRotation', 'egoHeading']]
        camera_config_df.filename = camera_config_df.filename.apply(lambda x: TEST_FILE_DIR + x)

        # 3. Decode filtered_frames and track
        self.optimize_decoding.optimize_tracking([camera_config_df.filename.tolist()])
        df = pd.read_csv(TEST_TRACK_FILE, sep=" ", header=None, 
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
        df.frame_idx = df.frame_idx.add(-1)
        
        obj_info = get_obj_trajectory(df, ego_config)
        facing_relative_filtered = facing_relative_check(obj_info, 0, ego_config)
        for obj_id in facing_relative_filtered:
            ### frame idx of current obj that satisfies the condition
            filtered_idx = facing_relative_filtered[obj_id]
            current_obj_filtered_frames = camera_config_df.filename.iloc[filtered_idx]
            current_obj_bboxes = obj_info[obj_id]['bbox']
            import cv2
            import numpy as np
            
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

optimizeIngestion().run_test()