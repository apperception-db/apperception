from apperception.database import database
from apperception.utils import F, join, import_pickle

from optimized_ingestion.payload import Payload
from optimized_ingestion.pipeline import Pipeline
from optimized_ingestion.stages.decode_frame.parallel_decode_frame import ParallelDecodeFrame
from optimized_ingestion.stages.detection_2d.yolo_detection import YoloDetection
from optimized_ingestion.stages.detection_estimation import DetectionEstimation
from optimized_ingestion.stages.tracking_2d.strongsort import StrongSORT
from optimized_ingestion.stages.detection_3d.from_2d_and_road import From2DAndRoad
from optimized_ingestion.stages.tracking_3d.from_2d_and_road import From2DAndRoad as From2DAndRoad_3d
from optimized_ingestion.utils.query_analyzer import PipelineConstructor



def construct_base_pipeline():
    pipeline = Pipeline()
    pipeline.add_filter(filter=ParallelDecodeFrame())
    pipeline.add_filter(filter=YoloDetection())

    pipeline.add_filter(filter=From2DAndRoad())
    pipeline.add_filter(filter=StrongSORT())  # 2 Frame p Second
    pipeline.add_filter(filter=From2DAndRoad_3d())
    
    return pipeline

def construct_pipeline(world, base=True):
    pipeline = construct_base_pipeline()
    if base:
        return pipeline
    pipeline.stages.insert(3, DetectionEstimation())  # 5 Frame p Second
    PipelineConstructor().add_pipeline(pipeline)(world.kwargs['predicate'])

def associate_detection_info(tracking_result, detection_info_meta):
    for detection_info in detection_info_meta[tracking_result.frame_idx]:
        if detection_info.detection_id == tracking_result.detection_id:
            return detection_info
        
def get_tracks(sortmeta, detection_estimation_meta, base):
    trajectories = {}
    for i in range(len(sortmeta)):
        frame = sortmeta[i]
        for obj_id, tracking_result in frame.items():
            if obj_id not in trajectories:
                trajectories[obj_id] = []
            associated_detection_info = associate_detection_info(
                tracking_result, detection_estimation_meta) if not base else detection_estimation_meta[i]
            trajectories[obj_id].append((tracking_result, associated_detection_info))

    for trajectory in trajectories.values():
        last = len(trajectory) - 1
        for i, t in enumerate(trajectory):
            if i > 0:
                t[0].prev = trajectory[i - 1][0]
            if i < last:
                t[0].next = trajectory[i + 1][0]
    return trajectories

def format_trajectory(video_name, obj_id, track, base):
    timestamps: List[str] = []
    pairs: List[Tuple[float, float, float]] = []
    itemHeadings: List[int] = []
    translations: List[Tuple[float, float, float]] = []
    road_types: List[str] = []
    roadpolygons: List[List[Tuple[float, float]]] = []

    for tracking_result_3d, detection_info in track:
        if detection_info:
            camera_id = detection_info.camera_id if base else detection_info.ego_config.camera_id
            object_type = tracking_result_3d.object_type
            timestamps.append(detection_info.timestamp)
            pairs.append(tracking_result_3d.point)
            itemHeadings.append(detection_info.segment_heading)
            translations.append(detection_info.ego_translation if base else detection_info.ego_config.ego_translation)
            road_types.append(detection_info.road_type)
            roadpolygons.append(None if base else detection_info.road_polygon_info.polygon)
    if not len(timestamps):
        return None
    return [video_name+'_obj_'+str(obj_id), camera_id, object_type, timestamps, pairs,
            itemHeadings, translations, road_types, roadpolygons]

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
    road_types: List[str],
    roadpolygon_list: List[List[Tuple[float, float]]]
):
    traj_centroids = []
    translations = []
    itemHeadings = []
    roadTypes = []
    # roadPolygons = []
    prevTimestamp = None
    for timestamp, current_point, curItemHeading, current_trans, cur_road_type, cur_roadpolygon in zip(
        postgres_timestamps, pairs, itemHeading_list, translation_list, road_types, roadpolygon_list
    ):
        if prevTimestamp == timestamp:
            continue
        prevTimestamp = timestamp

        # Construct trajectory
        traj_centroids.append(f"POINT Z ({join(current_point, ' ')})@{timestamp}")
        translations.append(f"POINT Z ({join(current_trans, ' ')})@{timestamp}")
        itemHeadings.append(f"{curItemHeading}@{timestamp}")
        roadTypes.append(f"{cur_road_type}@{timestamp}")
#         polygon_point = ', '.join(join(cur_point, ' ') for cur_point in list(
#             zip(*cur_roadpolygon.exterior.coords.xy)))
#         roadPolygons.append(f"Polygon (({polygon_point}))@{timestamp}")

    # Insert the item_trajectory separately
    insert_trajectory = f"""
    INSERT INTO Item_General_Trajectory (itemId, cameraId, objectType, roadTypes, trajCentroids,
    translations, itemHeadings)
    VALUES (
        '{item_id}',
        '{camera_id}',
        '{object_type}',
        ttext '{{[{', '.join(roadTypes)}]}}',
        tgeompoint '{{[{', '.join(traj_centroids)}]}}',
        tgeompoint '{{[{', '.join(translations)}]}}',
        tfloat '{{[{', '.join(itemHeadings)}]}}'
    );
    """

    database.execute(insert_trajectory)
    database._commit()

def process_pipeline(video_name, frames, pipeline, base):
    output = pipeline.run(Payload(frames)).__dict__
    metadata = output['metadata']
    if base:
        detection_estimation_meta = frames.camera_config
    else:
        detection_estimation_meta = metadata['DetectionEstimation']
    sortmeta = metadata['Tracking2D.StrongSORT']
    tracks = get_tracks(sortmeta, detection_estimation_meta, base)
    for obj_id, track in tracks.items():
        trajectory = format_trajectory(video_name, obj_id, track, base)
        if trajectory:
            print("Inserting trajectory")
            insert_trajectory(database, *trajectory)

