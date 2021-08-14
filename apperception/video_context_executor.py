from video_context import *
from video_util import *

import json

# TODO: Add checks for Nones 
class VideoContextExecutor:
    def __init__(self, conn, new_video_context:VideoContext=None):
        if new_video_context:
            self.context(new_video_context)
        self.conn = conn

    def context(self, video_context:VideoContext):
        self.current_context = video_context
        return self

    def visit(self):
        video_query = self.visit_world()
        return video_query

    def visit_world(self):
        # Query to store world in database 
        name, units = self.current_context.name, self.current_context.units
        world_sql = create_or_insert_world_table(self.conn, name, units)

        all_sqls = []
        cameras = self.current_context.camera_nodes
        if len(cameras) != 0:
            for c in cameras.values():
                camera_sql = self.visit_camera(c)
                all_sqls.append(camera_sql)
        return all_sqls

    def visit_camera(self, camera_node):
        world_name = self.current_context.name
        camera_sql = create_or_insert_camera_table(self.conn, world_name, camera_node)
        if camera_node.object_recognition is not None:
            self.visit_obj_rec(camera_node, camera_node.object_recognition)
        if self.current_context.tasm:
            video_data_to_tasm(camera_node.video_file, camera_node.metadata_id, self.current_context.tasm)
        return camera_sql

    def visit_obj_rec(self, camera_node, object_rec_node):
        cam_id = camera_node.cam_id
        lens = camera_node.lens
        video_file = camera_node.video_file

        start_time = self.current_context.start_time

        tracker = object_rec_node.tracker
        tracker_type = object_rec_node.tracker_type
        algo = object_rec_node.algo

        
        tracking_results = recognize(video_file, algo, tracker_type, tracker)
        add_recognized_objs(self.conn, lens, tracking_results, start_time)
        if self.current_context.tasm:
            metadata_to_tasm(tracking_results, camera_node.metadata_id, self.current_context.tasm)
        
    def execute(self):
        query = self.visit()

