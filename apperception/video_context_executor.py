from video_context import VideoContext
from video_util import create_or_insert_world_table, video_data_to_tasm, metadata_to_tasm
from scenic_util import create_or_insert_camera_table, recognize, add_recognized_objs

import json

class VideoContextExecutor:
    def __init__(self, conn, new_video_context:VideoContext=None, tasm=None):
        if new_video_context:
            self.context(new_video_context)
        self.conn = conn
        self.tasm = tasm

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
        if self.tasm:
            video_data_to_tasm(camera_node, camera_node.metadata_id, self.tasm)
        return camera_sql

    def visit_obj_rec(self, camera_node, object_rec_node):
        cam_id = camera_node.scenic_scene_name

        start_time = self.current_context.start_time

        tracking_results = recognize(cam_id, object_rec_node.sample_data, object_rec_node.annotation)
        add_recognized_objs(self.conn, tracking_results, start_time)
        if self.tasm:
            metadata_to_tasm(tracking_results, camera_node.metadata_id, self.tasm)
        
    def execute(self):
        query = self.visit()

