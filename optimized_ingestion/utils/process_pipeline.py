from apperception.database import database
from apperception.utils import join

import math
import time

from optimized_ingestion.payload import Payload
from optimized_ingestion.pipeline import Pipeline
from optimized_ingestion.stages.decode_frame.parallel_decode_frame import \
    ParallelDecodeFrame
from optimized_ingestion.stages.detection_2d.yolo_detection import \
    YoloDetection
from optimized_ingestion.stages.detection_3d.from_detection_2d_and_road import \
    FromDetection2DAndRoad
from optimized_ingestion.stages.detection_estimation import DetectionEstimation
from optimized_ingestion.stages.segment_trajectory.from_tracking_3d import \
    FromTracking3D
from optimized_ingestion.stages.tracking_2d.strongsort import StrongSORT
from optimized_ingestion.stages.tracking_3d.from_tracking_2d_and_road import \
    FromTracking2DAndRoad as From2DAndRoad_3d
from optimized_ingestion.utils.query_analyzer import PipelineConstructor


def construct_base_pipeline():
    pipeline = Pipeline()
    pipeline.add_filter(filter=ParallelDecodeFrame())
    pipeline.add_filter(filter=YoloDetection())

    pipeline.add_filter(filter=FromDetection2DAndRoad())
    pipeline.add_filter(filter=StrongSORT())  # 2 Frame p Second
    pipeline.add_filter(filter=From2DAndRoad_3d())
    pipeline.add_filter(filter=FromTracking3D())

    return pipeline


def construct_pipeline(world, base):
    pipeline = construct_base_pipeline()
    if base:
        return pipeline
    pipeline.stages.insert(3, DetectionEstimation())
    if world.kwargs.get('predicate'):
        PipelineConstructor().add_pipeline(pipeline)(world.kwargs.get('predicate'))
    return pipeline


def associate_detection_info(tracking_result, detection_info_meta):
    for detection_info in detection_info_meta[tracking_result.frame_idx]:
        if detection_info.detection_id == tracking_result.detection_id:
            return detection_info


def associate_segment_mapping(tracking_result, segment_mapping_meta):
    return segment_mapping_meta[tracking_result.frame_idx][tracking_result.object_id]


def get_tracks(sortmeta, ego_meta, segment_mapping_meta, base):
    trajectories = {}
    for i in range(len(sortmeta)):
        frame = sortmeta[i]
        for obj_id, tracking_result in frame.items():
            if obj_id not in trajectories:
                trajectories[obj_id] = []
            associated_ego_info = ego_meta[i]
            associated_segment_mapping = associate_segment_mapping(tracking_result, segment_mapping_meta)
            trajectories[obj_id].append((tracking_result, associated_ego_info, associated_segment_mapping))

    for trajectory in trajectories.values():
        last = len(trajectory) - 1
        for i, t in enumerate(trajectory):
            if i > 0:
                t[0].prev = trajectory[i - 1][0]
            if i < last:
                t[0].next = trajectory[i + 1][0]
    return trajectories


def infer_heading(curItemHeading, prevPoint, current_point):
    if curItemHeading is not None:
        return math.degrees(curItemHeading)
    if prevPoint is None:
        return None
    x1, y1, z1 = prevPoint
    x2, y2, z2 = current_point
    return int(math.degrees(math.atan2(y2 - y1, x2 - x1)) - 90)


def format_trajectory(video_name, obj_id, track, base):
    timestamps: List[str] = []
    pairs: List[Tuple[float, float, float]] = []
    itemHeadings: List[int] = []
    translations: List[Tuple[float, float, float]] = []
    # road_types: List[str] = []
    # roadpolygons: List[List[Tuple[float, float]]] = []
    for tracking_result_3d, ego_info, segment_mapping in track:
        if ego_info:
            camera_id = ego_info.camera_id
            object_type = tracking_result_3d.object_type
            timestamps.append(ego_info.timestamp)
            pairs.append(tracking_result_3d.point)
            if (segment_mapping.segment_type == 'intersection'):
                itemHeadings.append(None)
            else:
                itemHeadings.append(segment_mapping.segment_heading)
            translations.append(ego_info.ego_translation)
            # road_types.append(segment_mapping.road_polygon_info.road_type if base else detection_info.road_type)
            # roadpolygons.append(None if base else detection_info.road_polygon_info.polygon)
    if not len(timestamps):
        return None
    # if obj_id in investigation_ids:
    #     print(f"pairs for obj {obj_id}:", [(e[0], e[1]) for e in pairs])
    #     print(f"itemHeadings for obj {obj_id}:", itemHeadings)

    return [video_name + '_obj_' + str(obj_id), camera_id, object_type, timestamps, pairs,
            itemHeadings, translations]


from typing import List, Tuple


def insert_trajectory(
    database,
    item_id: str,
    camera_id: str,
    object_type: str,
    postgres_timestamps: List[str],
    pairs: List[Tuple[float, float, float]],
    itemHeading_list: List[int],
    translation_list: List[Tuple[float, float, float]],
    # road_types: List[str],
    # roadpolygon_list: List[List[Tuple[float, float]]]
):
    traj_centroids = []
    translations = []
    itemHeadings = []
    prevTimestamp = None
    prevPoint = None
    for timestamp, current_point, curItemHeading, current_trans in zip(
        postgres_timestamps, pairs, itemHeading_list, translation_list
    ):
        if prevTimestamp == timestamp:
            continue
        prevTimestamp = timestamp

        # Construct trajectory
        traj_centroids.append(f"POINT Z ({join(current_point, ' ')})@{timestamp}")
        translations.append(f"POINT Z ({join(current_point, ' ')})@{timestamp}")
        curItemHeading = infer_heading(curItemHeading, prevPoint, current_point)
        if curItemHeading is not None:
            itemHeadings.append(f"{curItemHeading}@{timestamp}")
        # roadTypes.append(f"{cur_road_type}@{timestamp}")
#         polygon_point = ', '.join(join(cur_point, ' ') for cur_point in list(
#             zip(*cur_roadpolygon.exterior.coords.xy)))
#         roadPolygons.append(f"Polygon (({polygon_point}))@{timestamp}")
        prevPoint = current_point

    # Insert the item_trajectory separately
    item_headings = f"tfloat 'Interp=Stepwise;{{[{', '.join(itemHeadings)}]}}'" if itemHeadings else "null"
    insert_trajectory = f"""
    INSERT INTO Item_General_Trajectory (itemId, cameraId, objectType, trajCentroids,
    translations, itemHeadings)
    VALUES (
        '{item_id}',
        '{camera_id}',
        '{object_type}',
        tgeompoint '{{[{', '.join(traj_centroids)}]}}',
        tgeompoint '{{[{', '.join(translations)}]}}',
        {item_headings}
    );
    """

    database.execute(insert_trajectory)
    database._commit()


def process_pipeline(video_name, frames, pipeline, base):
    output = pipeline.run(Payload(frames)).__dict__
    metadata = output['metadata']
    ego_meta = frames.interpolated_frames
    sortmeta = metadata['Tracking3D.From2DAndRoad']
    segment_trajectory_mapping = metadata['SegmentTrajectory.FromTracking3D']
    tracks = get_tracks(sortmeta, ego_meta, segment_trajectory_mapping, base)
    start = time.time()
    for obj_id, track in tracks.items():
        trajectory = format_trajectory(video_name, obj_id, track, base)

        if trajectory:
            # print("Inserting trajectory")
            insert_trajectory(database, *trajectory)
    trajectory_ingestion_time = time.time() - start
    print("Time taken to insert trajectories:", trajectory_ingestion_time)
