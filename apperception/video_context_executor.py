from typing import Any, List, Set

from bounding_box import BoundingBox
from scenic_util import (add_recognized_objs, create_or_insert_camera_table,
                         recognize)
from video_context import Camera, VideoContext
from video_util import (create_or_insert_world_table, metadata_to_tasm,
                        video_data_to_tasm)


class VideoContextExecutor:
    # TODO: Add checks for Nones

    def __init__(self, conn: Any, new_video_context: VideoContext = None, tasm=None):
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

        all_sqls = []
        cameras = self.current_context.camera_nodes
        if len(cameras) != 0:
            for c in cameras.values():
                camera_sql = self.visit_camera(c)
                all_sqls.append(camera_sql)
        return all_sqls

    def visit_camera(self, camera_node: Camera):
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

        tracking_results = recognize(
            cam_id, object_rec_node.sample_data, object_rec_node.annotation
        )
        add_recognized_objs(self.conn, tracking_results, start_time)
        if self.tasm:
            metadata_to_tasm(tracking_results, camera_node.metadata_id, self.tasm)

    def execute(self):
        query = self.visit()


def is_area_recognized(area: BoundingBox, recognized: Set[BoundingBox]):
    for other in recognized:
        if area.is_in(other):
            return True
    return False


def to_recognize_whole_frame(recognition_areas: List[BoundingBox]):
    if len(recognition_areas) == 0:
        return False

    area = recognition_areas[0]
    if area.is_whole_frame() or (
        area.x1 == 0 and area.y1 == 0 and area.x2 == 100 and area.y2 == 100
    ):
        return True

    (y1, x1), (y2, x2) = recognition_areas[0].to_tuples()
    for area in recognition_areas[1:]:
        if area.is_whole_frame():
            return True

        x1 = min(area.x1, x1, 0)
        x2 = max(area.x2, x2, 100)
        y1 = min(area.y1, y1, 0)
        y2 = max(area.y2, y2, 100)

    return (x2 - x1) * (y2 - y1) >= 100 * 100 / 2.0
