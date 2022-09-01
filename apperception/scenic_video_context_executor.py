from scenic_context import *
from scenic_util import *

import json

class ScenicVideoContextExecutor:
    def __init__(self, conn, new_video_context:ScenicVideoContext=None, tasm=None):
        if new_video_context:
            self.context(new_video_context)
        self.conn = conn
        self.tasm = tasm

    def context(self, video_context:ScenicVideoContext):
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
        camera_sql = create_or_insert_scenic_camera_table(self.conn, world_name, camera_node)
        if camera_node.object_recognition is not None:
            self.visit_obj_rec(camera_node, camera_node.object_recognition)
        if self.tasm:
            video_data_to_tasm(camera_node, camera_node.metadata_id, self.tasm)
        return camera_sql

    def visit_obj_rec(self, camera_node, object_rec_node):
        cam_id = camera_node.scenic_scene_name

        start_time = self.current_context.start_time

        tracking_results = scenic_recognize(cam_id, object_rec_node.sample_data, object_rec_node.annotation)
        add_scenic_recognized_objs(self.conn, cam_id, tracking_results, start_time)
        if self.tasm:
            metadata_to_tasm(tracking_results, camera_node.metadata_id, self.tasm)
        
    def execute(self):
        query = self.visit()

