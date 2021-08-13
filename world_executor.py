from metadata_context_executor import *
from metadata_context import *
from video_context_executor import *
from video_util import *
import numpy as np

class WorldExecutor:
    def __init__(self, world):
        if world:
            self.create_world(world)
    
    def connect_db(self, 
        host='localhost',
        user=None,
        password=None,
        port=5432,
        database_name=None):

        self.conn = psycopg2.connect(database=database_name, user=user, password=password, host=host, port=port)

    def create_world(self, world):
        self.curr_world = world
        return self
    
    def get_camera(self, cam_id = []):
        assert self.curr_world, self.conn
        cameras = fetch_camera(self.conn, self.curr_world.get_name(), cam_id)
        ### each camera appear like:
        ### (cameraId, ratio, origin3d, focalpoints2d, fov, skev_factor)
        
        return cameras
    
    def tasm_get_video(self, metadata_results):
        ### Get the metadata context executing query text, let tasm get video call it
        ### the tasm would execute the query to get the ids, bboxes and timestamps
        ### then it can use these to tile the video and get it
        cam_nodes = self.curr_world.get_video_cams
        tasm = self.curr_world.fetch_tasm()
        all_selection = {}
        for cam_node in cam_nodes:
            current_metadata_identifier = cam_node.metadata_id
            current_video_file = cam_nodes.video_file
            tasm.activate_regret_based_tiling(current_video_file, current_metadata_identifier)
            for label, timestamps in metadata_results.items():
                selection = tasm.select_tiles(current_video_file, current_metadata_identifier, label, timestamps[0], timestamps[-1])
                tasm.retile_based_on_regret(current_video_file, current_metadata_identifier)
            all_selection.update(cam_node, selection)
        return all_selection
    
    def get_video(self, metadata_results):
        start_time = self.curr_world.VideoContext.start_time
        # print("Start time is", start_time)
        ### The cam nodes are raw data from the database
        ### TODO: I forget why we used the data from the db instead of directly fetch
        ### from the world
        cam_nodes = self.curr_world.get_video_cams
        video_files = []
        ### TODO: Integration to TASM
        for i in range(len(cam_nodes)):
            cam_id, ratio, cam_x, cam_y, cam_z, focal_x, focal_y, fov, skew_factor = cam_nodes[i]
            cam_video_file = self.curr_world.VideoContext.camera_nodes[cam_id].video_file

            transform_matrix = create_transform_matrix(focal_x, focal_y, cam_x, cam_y, skew_factor)
            
            for item_id, vals in metadata_results.items():
                world_coords, timestamps = vals
                # print("timestamps are", timestamps)
                world_coords = reformat_fetched_world_coords(world_coords)

                cam_coords = world_to_pixel(world_coords, transform_matrix)
                # print("is it able to convert to pixels")
                # print(cam_coords)
                vid_times = convert_datetime_to_frame_num(start_time, timestamps)
                # print(vid_times)

                vid_fname = self.curr_world.VideoContext.camera_nodes[cam_id].metadata_id + cam_id + item_id + '.mp4'
                # print(vid_fname)

                get_video_roi(vid_fname, cam_video_file, cam_coords, vid_times) 
                # print("is it able to produce video?")
                video_files.append(vid_fname)
                
        return video_files
        
    def execute(self):
        # Edit logic for execution here through checks of whether VideoContext or MetadataContext is being used 
        video_executor = VideoContextExecutor(self.conn, self.curr_world.VideoContext)
        video_executor.execute()

        if self.curr_world.MetadataContext.scan.view == None:
            return

        if self.curr_world.GetVideo:
            metadata_executor = MetadataContextExecutor(self.conn, self.curr_world.MetadataContext.get_columns(primarykey, time))
            metadata_results = video_fetch_reformat(metadata_executor.execute())
            # return self.get_video(metadata_results)
            return self.tasm_get_video(metadata_results)

        metadata_executor = MetadataContextExecutor(self.conn, self.curr_world.MetadataContext)
        return metadata_executor.execute()

def create_transform_matrix(focal_x, focal_y, cam_x, cam_y, skew_factor):
    alpha = skew_factor

    transform = np.array([[focal_x, alpha, cam_x, 0], 
                                    [0, focal_y, cam_y, 0],
                                    [0, 0, 1, 0]
                                   ])

    return transform

def reformat_fetched_world_coords(world_coords):
    return np.array(world_coords)

def world_to_pixel(world_coords, transform):
    tl_x, tl_y, tl_z, br_x, br_y, br_z = world_coords.T

    tl_world_pixels = np.array([tl_x, tl_y, tl_z, np.ones(len(tl_x))])
    tl_vid_coords = transform @ tl_world_pixels

    br_world_pixels = np.array([br_x, br_y, br_z, np.ones(len(br_x))])
    br_vid_coords = transform @ br_world_pixels

    return np.stack((tl_vid_coords[0], tl_vid_coords[1], br_vid_coords[0], br_vid_coords[1]), axis=0)

def video_fetch_reformat(fetched_meta):
    result = {}
    for meta in fetched_meta:
        item_id, timestamp = meta[0], meta[1]
        if item_id in result:
            result[item_id]['tracked_cnt'].append(timestamp)
        else:
            result[item_id] = {'tracked_cnt':[timestamp]}

    return result
        