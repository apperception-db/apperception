from typing import Dict, List

from video_context import Camera, ObjectRecognition, VideoContext
from video_util import (add_recognized_objs, create_or_insert_camera_table,
                        create_or_insert_world_table, metadata_to_tasm,
                        recognize, video_data_to_tasm)

recognized_camera: Dict[str, bool] = {}


class VideoContextExecutor:
    # TODO: Add checks for Nones
    def __init__(self, conn, new_video_context: VideoContext = None, tasm=None):
        if new_video_context:
            self.context(new_video_context)
        self.conn = conn
        self.tasm = tasm

    def context(self, video_context: VideoContext):
        self.current_context = video_context
        return self

    def visit(self):
        video_query = self.visit_world()
        return video_query

    def visit_world(self):
        # Query to store world in database
        name, units = self.current_context.name, self.current_context.units
        world_sql = create_or_insert_world_table(self.conn, name, units)

        all_sqls: List[str] = []
        cameras = self.current_context.camera_nodes
        if len(cameras) != 0:
            for c in cameras.values():
                cam_id = c.cam_id
                if not recognized_camera.get(cam_id, False):
                    camera_sql = self.visit_camera(c)
                    all_sqls.append(camera_sql)
                    recognized_camera[cam_id] = True
        return all_sqls

    def visit_camera(self, camera_node: Camera):
        world_name = self.current_context.name
        camera_sql = create_or_insert_camera_table(self.conn, world_name, camera_node)
        if camera_node.object_recognition is not None:
            self.visit_obj_rec(camera_node, camera_node.object_recognition)
        if self.tasm:
            video_data_to_tasm(camera_node.video_file, camera_node.metadata_id, self.tasm)
        return camera_sql

    def visit_obj_rec(self, camera_node: Camera, object_rec_node: ObjectRecognition):
        lens = camera_node.lens
        video_file = camera_node.video_file

        start_time = self.current_context.start_time

        tracker = object_rec_node.tracker
        tracker_type = object_rec_node.tracker_type
        algo = object_rec_node.algo

        tracking_results = recognize(video_file, algo, tracker_type, tracker)
        add_recognized_objs(self.conn, lens, tracking_results, start_time)
        if self.tasm:
            metadata_to_tasm(tracking_results, camera_node.metadata_id, self.tasm)

    def execute(self):
        query = self.visit()
