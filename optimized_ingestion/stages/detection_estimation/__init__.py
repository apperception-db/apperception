import logging
import time
from typing import Callable, List, Tuple

import torch
from bitarray import bitarray
from psycopg2 import sql

from apperception.database import database

from ...camera_config import CameraConfig
from ...payload import Payload
from ...types import DetectionId
from ...video import Video
from ..detection_2d.detection_2d import Detection2D
from ..detection_3d import Detection3D
from ..in_view.in_view import get_views
from ..stage import Stage
from .detection_estimation import (
    DetectionInfo,
    SamplePlan,
    construct_all_detection_info,
    generate_sample_plan,
    obj_detection,
)
from .utils import get_ego_avg_speed, trajectory_3d

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


DetectionEstimationMetadatum = List[DetectionInfo]


class DetectionEstimation(Stage[DetectionEstimationMetadatum]):

    def __init__(self, predicate: "Callable[[DetectionInfo], bool]" = lambda _: True):
        self.predicates = [predicate]
        super(DetectionEstimation, self).__init__()

    def add_filter(self, predicate: "Callable[[DetectionInfo], bool]"):
        self.predicates.append(predicate)

    def _run(self, payload: "Payload"):
        if Detection2D.get(payload) is None:
            raise Exception()

        keep = bitarray(len(payload.video))
        keep[:] = 1

        ego_trajectory = [trajectory_3d(f.ego_translation, f.timestamp) for f in payload.video]
        ego_speed = get_ego_avg_speed(ego_trajectory)
        logger.info(f"ego_speed: {ego_speed}")
        if ego_speed < 2:
            return keep, {DetectionEstimation.classname(): [[]] * len(keep)}

        ego_views = get_ego_views(payload)
        assert ego_views is not None

        skipped_frame_num = []
        next_frame_num = 0
        action_type_counts = {}
        start_time = time.time()
        total_detection_time = 0
        total_sample_plan_time = 0
        dets = Detection3D.get(payload)
        assert dets is not None, [*payload.metadata.keys()]
        metadata: "list[DetectionEstimationMetadatum]" = []
        start_time = time.time()
        investigation_frame_nums = []
        for i in Stage.tqdm(range(len(payload.video) - 1)):
            current_ego_config = payload.video[i]
            if i != next_frame_num:
                skipped_frame_num.append(i)
                metadata.append([])
                continue
            next_frame_num = i + 1
            start_detection_time = time.time()
            det, _, dids = dets[i]
            if i == 195:
                all_3d_points = []
                for de, di in zip(det, dids):
                    bbox = de[:4]
                    bbox3d = de[6:12]
                    left3d, right3d = bbox3d[:3], bbox3d[3:]
                    car_loc3d = tuple(map(float, (left3d + right3d) / 2))
                    all_3d_points.append(car_loc3d)
            logger.info(f'current frame num {i}')
            all_detection_info = construct_estimated_all_detection_info(det, dids, current_ego_config, ego_trajectory)
            total_detection_time += time.time() - start_detection_time
            all_detection_info_pruned, det = prune_detection(all_detection_info, det, self.predicates)
            # assert len(all_detection_info_pruned) == len(det), (len(all_detection_info_pruned), len(det))
            if len(det) == 0:
                skipped_frame_num.append(i)
                metadata.append([])
                continue
            start_generate_sample_plan = time.time()
            next_sample_plan, _ = generate_sample_plan_once(payload.video, current_ego_config, next_frame_num, all_detection_info=all_detection_info_pruned)
            total_sample_plan_time += time.time() - start_generate_sample_plan
            next_action_type = next_sample_plan.get_action_type()
            if next_action_type not in action_type_counts:
                action_type_counts[next_action_type] = 1
            else:
                action_type_counts[next_action_type] += 1
            next_frame_num = next_sample_plan.get_next_frame_num(next_frame_num)
            investigation_frame_nums.append([i, next_action_type])
            if next_action_type:
                investigation_frame_nums[-1].extend([next_sample_plan.action.target_obj_bbox])
            else:
                investigation_frame_nums[-1].extend([None])
            metadata.append(all_detection_info)

        # TODO: ignore the last frame ->
        metadata.append([])
        skipped_frame_num.append(len(payload.video) - 1)

        #     times.append([t2 - t1 for t1, t2 in zip(t[:-1], t[1:])])
        # logger.info(np.array(times).sum(axis=0))
        logger.info(f"sorted_ego_config_length {len(payload.video)}")
        logger.info(f"investigation_frame_nums {investigation_frame_nums}")
        logger.info(f"number of skipped {len(skipped_frame_num)}")
        logger.info(action_type_counts)
        total_run_time = time.time() - start_time
        logger.info(f"total_run_time {total_run_time}")
        logger.info(f"total_detection_time {total_detection_time}")
        logger.info(f"total_generate_sample_plan_time {total_sample_plan_time}")

        for f in skipped_frame_num:
            keep[f] = 0

        return keep, {DetectionEstimation.classname(): metadata}


def get_ego_views(payload: "Payload"):
    indices, view_areas = get_views(payload, distance=100, skip=False)
    views_raw = database.execute(sql.SQL("""
    SELECT index, ST_ConvexHull(points)
    FROM UNNEST (
        {view_areas},
        {indices}::int[]
    ) AS ViewArea(points, index)
    """).format(
        view_areas=sql.Literal(view_areas),
        indices=sql.Literal(indices),
    ))
    assert len(views_raw) == len(payload.video), (len(views_raw), len(payload.video))
    views = [None for _ in range(len(payload.video))]
    for idx, view in views_raw:
        views[idx] = view
    return views


def prune_detection(
    detection_info: "list[DetectionInfo]",
    det: "torch.Tensor",
    predicates: "List[Callable[[DetectionInfo], bool]]"
):
    # TODO (fge): this is a hack before fixing the mapping between det and detection_info
    return detection_info, det
    if len(detection_info) == 0:
        return detection_info, det
    pruned_detection_info: "list[DetectionInfo]" = []
    pruned_det: "list[torch.Tensor]" = []
    for d, di in zip(det, detection_info):
        if all([predicate(di) for predicate in predicates]):
            pruned_detection_info.append(di)
            pruned_det.append(d)
    logger.info("length before pruning", len(det))
    logger.info("length after pruning", len(pruned_det))
    return pruned_detection_info, pruned_det


def generate_sample_plan_once(
    video: "Video",
    ego_config: "CameraConfig",
    next_frame_num: "int",
    car_loc3d=None,
    target_car_detection=None,
    all_detection_info: "List[DetectionInfo] | None" = None
) -> "Tuple[SamplePlan, None]":
    assert all_detection_info is not None
    next_sample_plan = generate_sample_plan(video, next_frame_num, all_detection_info, 50)
    # next_frame = None
    next_sample_frame_info = next_sample_plan.get_next_sample_frame_info()
    if next_sample_frame_info:
        next_sample_frame_name, next_sample_frame_num, _ = next_sample_frame_info
        logger.info(f"next frame name {next_sample_frame_name}")
        logger.info(f"next frame num {next_sample_frame_num}")
        logger.info(f"Action {next_sample_plan.action}")
        # TODO: should not read next frame -> get the next frame from frames.pickle
        # next_frame = cv2.imread(test_img_base_dir+next_sample_frame_name)
        # cv2.imshow("next_frame", next_frame)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    return next_sample_plan, None


def construct_estimated_all_detection_info(
    detections: "torch.Tensor",
    detection_ids: "list[DetectionId]",
    ego_config: "CameraConfig",
    ego_trajectory: "list[trajectory_3d]",
) -> "list[DetectionInfo]":
    all_detections = []
    check_detections = []
    if len(detection_ids) > 0:
        print(detection_ids[0].frame_idx)
    for det, did in zip(detections, detection_ids):
        bbox = det[:4]
        # conf = det[4]
        # obj_cls = det[5]
        # bbox3d_from_camera = det[12:18]
        bbox3d = det[6:12]
        x, y, x2, y2 = list(map(int, bbox))
        w = x2 - x
        h = y2 - y
        car_loc2d = (x + w // 2, y + h // 2)
        car_bbox2d = ((x, y), (x2, y2))
        left3d, right3d = bbox3d[:3], bbox3d[3:]
        car_loc3d = tuple(map(float, (left3d + right3d) / 2))
        assert len(car_loc3d) == 3
        car_bbox3d = (tuple(map(float, left3d)), tuple(map(float, right3d)))
        all_detections.append(obj_detection(
            did,
            car_loc3d,
            car_loc2d,
            car_bbox3d,
            car_bbox2d)
        )
    all_detection_info = construct_all_detection_info(ego_config, ego_trajectory, all_detections)
    for di in all_detection_info:
        if di.detection_id.frame_idx == 207 or di.detection_id.frame_idx == 190:
            # print(di.road_polygon_info.id)
            x, y = di.car_bbox2d[0]
            x_w, y_h = di.car_bbox2d[1]
            check_detections.append([di.detection_id.obj_order, x, y, x_w, y_h,
                                     di.road_type, di.road_polygon_info.polygon2d.exterior.coords.xy, di.ego_config.filename])
    if len(check_detections) > 0:
        print(f"check_detections {check_detections}")
    return all_detection_info
