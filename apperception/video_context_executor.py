from typing import Any, Dict, List, Optional, Set, Tuple

from video_context import Camera, ObjectRecognition, VideoContext
from video_util import (add_recognized_objs,
                        create_or_insert_camera_table, get_video_dimension,
                        create_or_insert_world_table, metadata_to_tasm,
                        recognize, video_data_to_tasm)
from bounding_box import BoundingBox, WHOLE_FRAME

recognized_areas: Dict[str, Set[BoundingBox]] = {}
visited_camera: Set[str] = set()
created_world: Set[str] = set()


class VideoContextExecutor:
    # TODO: Add checks for Nones

    def __init__(
        self, conn: Any, new_video_context: VideoContext = None, tasm=None
    ):
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

        if name not in created_world:
            world_sql = create_or_insert_world_table(self.conn, name, units)
            created_world.add(name)

        all_sqls: List[str] = []
        cameras = self.current_context.camera_nodes
        for c in cameras.values():
            camera_sql = self.visit_camera(c)
            all_sqls.append(camera_sql)
        return all_sqls

    def visit_camera(self, camera_node: Camera):
        world_name = self.current_context.name
        camera_sql = ""

        if camera_node.cam_id not in visited_camera:
            # Only insert camera once
            camera_node.dimension = get_video_dimension(camera_node.video_file)
            camera_sql = create_or_insert_camera_table(self.conn, world_name, camera_node)
            visited_camera.add(camera_node.cam_id)
            recognized_areas[camera_node.cam_id] = set()

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

        recognition_areas = object_rec_node.recognition_areas

        if to_recognize_whole_frame(recognition_areas, camera_node.dimension):
            tracking_results = recognize(video_file, algo, tracker_type, tracker)
            # TODO: @mick remove other recognized object of this camera
            add_recognized_objs(self.conn, lens, tracking_results, start_time)
            recognized_areas[camera_node.cam_id] = {WHOLE_FRAME}
        else:
            recognition_areas = sorted(recognition_areas, key=lambda a : a.area, reverse=True)
            for recognition_area in recognition_areas:
                if is_area_recognized(recognition_area, recognized_areas[camera_node.cam_id]):
                    continue

                tracking_results = recognize(video_file, algo, tracker_type, tracker, recognition_area)
                add_recognized_objs(self.conn, lens, tracking_results, start_time)
                # TODO: should remove recognized objects in overlapped areas

                recognized_areas[camera_node.cam_id].add(recognition_area)

        if self.tasm:
            metadata_to_tasm(tracking_results, camera_node.metadata_id, self.tasm)

    def execute(self):
        query = self.visit()


def is_area_recognized(area: BoundingBox, recognized: Set[BoundingBox]):
    for other in recognized:
        if area.is_in(other):
            return True
    return False


def to_recognize_whole_frame(recognition_areas: List[BoundingBox], dimension: Optional[Tuple[int, int]] = None):
    if len(recognition_areas) == 0:
        return False

    if recognition_areas[0].is_whole_frame() or dimension is None:
        return True

    (x1, y1), (x2, y2) = recognition_areas[0].to_tuples()
    for area in recognition_areas[1:]:
        if area.is_whole_frame():
            return True

        x1 = min(area.x1, x1)
        x2 = min(area.x2, x2)
        y1 = min(area.y1, y1)
        y2 = min(area.y2, y2)

    return (x2 - x1) * (y2 - y2) >= dimension[0] * dimension[1] / 2.
