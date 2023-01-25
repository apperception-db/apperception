import logging
import time
import torch
from bitarray import bitarray
from tqdm import tqdm
from typing import Callable, List, Tuple

from ...camera_config import CameraConfig
from ...payload import Payload
from ...types import DetectionId
from ...video import Video
from ..detection_2d.detection_2d import Detection2D
from ..detection_3d import Detection3D
from ..stage import Stage
from .detection_estimation import (DetectionInfo, SamplePlan,
                                   construct_all_detection_info,
                                   generate_sample_plan, obj_detection)
from .segment_mapping import CameraPolygonMapping, map_imgsegment_roadsegment
from .utils import trajectory_3d

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARN)


DetectionEstimationMetadatum = List[DetectionInfo]


class DetectionEstimation(Stage[DetectionEstimationMetadatum]):
    def _run(self, payload: "Payload"):
        if Detection2D.get(payload) is None:
            raise Exception()

        ego_trajectory = [trajectory_3d(f.ego_translation, f.timestamp) for f in payload.video]

        skipped_frame_num = []
        next_frame_num = 0
        action_type_counts = {}
        start_time = time.time()
        total_detection_time = 0
        total_sample_plan_time = 0
        # times = []
        dets = Detection3D.get(payload)
        assert dets is not None, [*payload.metadata.keys()]
        metadata: "list[DetectionEstimationMetadatum]" = []
        start_time = time.time()
        mapping_time = 0
        for i in tqdm(range(len(payload.video) - 1)):
            current_ego_config = payload.video[i]
            if i != next_frame_num:
                skipped_frame_num.append(i)
                metadata.append([])
                continue
            next_frame_num = i + 1
            start_mapping_time = time.time()
            cam_polygon_mapping = map_imgsegment_roadsegment(current_ego_config)
            mapping_time += time.time() - start_mapping_time
            logger.info(f"mapping length {len(cam_polygon_mapping)}")
            start_detection_time = time.time()
            det, _ = dets[i]
            all_detection_info = construct_estimated_all_detection_info(det, cam_polygon_mapping, current_ego_config, ego_trajectory, i)
            all_detection_info, det = prune_detection(all_detection_info, det)
            assert len(all_detection_info) == len(det), (len(all_detection_info), len(det))
            if len(all_detection_info) == 0:
                skipped_frame_num.append(i)
                metadata.append([])
                continue
            total_detection_time += time.time() - start_detection_time
            start_generate_sample_plan = time.time()
            next_sample_plan, _ = generate_sample_plan_once(payload.video, current_ego_config, cam_polygon_mapping, next_frame_num, all_detection_info=all_detection_info)
            total_sample_plan_time += time.time() - start_generate_sample_plan
            next_action_type = next_sample_plan.get_action_type()
            if next_action_type not in action_type_counts:
                action_type_counts[next_action_type] = 1
            else:
                action_type_counts[next_action_type] += 1
            next_frame_num = next_sample_plan.get_next_frame_num(next_frame_num)
            metadata.append(all_detection_info)

        # TODO: ignore the last frame ->
        metadata.append([])
        skipped_frame_num.append(len(payload.video) - 1)

        #     times.append([t2 - t1 for t1, t2 in zip(t[:-1], t[1:])])
        # logger.info(np.array(times).sum(axis=0))
        logger.info(f"sorted_ego_config_length {len(payload.video)}")
        logger.info(f"number of skipped {len(skipped_frame_num)}")
        logger.info(skipped_frame_num)
        logger.info(action_type_counts)
        total_run_time = time.time() - start_time
        num_runs = len(payload.video) - len(skipped_frame_num)
        logger.info(f"total_run_time {total_run_time}")
        logger.info(f"avg run time {total_run_time/num_runs}")
        logger.info(f"total_detection_time {total_detection_time}")
        logger.info(f"avg detection time {total_detection_time/num_runs}")
        logger.info(f"total_generate_sample_plan_time {total_sample_plan_time}")
        logger.info(f"avg generate_sample_plan time {total_sample_plan_time/num_runs}")
        logger.info(f"total_mapping_time {mapping_time}")
        logger.info(f"avg mapping time {mapping_time/num_runs}")

        keep = bitarray(len(payload.video))
        keep[:] = 1
        for f in skipped_frame_num:
            keep[f] = 0

        return keep, {DetectionEstimation.classname(): metadata}


def prune_detection(
    detection_info: "list[DetectionInfo]",
    det: "torch.Tensor",
    predicate: "Callable[[DetectionInfo], bool]" = lambda x: x.road_type == "intersection"
):
    pruned_detection_info: "list[DetectionInfo]" = []
    pruned_det: "list[torch.Tensor]" = []
    for d, di in zip(det, detection_info):
        if predicate(di):
            pruned_detection_info.append(di)
            pruned_det.append(d)
    logger.info("length before pruning", len(det))
    logger.info("length after pruning", len(pruned_det))
    return pruned_detection_info, pruned_det


def generate_sample_plan_once(
    video: "Video",
    ego_config: "CameraConfig",
    mapping: "List[CameraPolygonMapping]",
    next_frame_num: "int",
    car_loc3d=None,
    target_car_detection=None,
    all_detection_info: "List[DetectionInfo] | None" = None
) -> "Tuple[SamplePlan, None]":
    # if all_detection_info is None:
    #     assert target_car_detection and car_loc3d
    #     x,y,w,h = list(map(int, target_car_detection))
    #     car_loc2d = (x, y+h//2)
    #     car_bbox2d = (x-w//2,y-h//2,x+w//2,y+h//2)
    #     car_bbox3d = None
    #     all_detections = []
    #     all_detections.append(obj_detection('car_1', car_loc3d, car_loc2d, car_bbox3d, car_bbox2d))
    #     all_detection_info = construct_all_detection_info(cam_segment_mapping, ego_trajectory, ego_config, all_detections)
    assert all_detection_info is not None
    if all_detection_info:
        logger.info(all_detection_info[0].road_type)
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
    cam_polygon_mapping: "List[CameraPolygonMapping]",
    ego_config: "CameraConfig",
    ego_trajectory: "List[trajectory_3d]",
    frame_idx: int,
) -> "List[DetectionInfo]":
    all_detections = []
    for i, det in enumerate(detections):
        bbox = det[:4]
        # conf = det[4]
        # obj_cls = det[5]
        # bbox3d_from_camera = det[12:18]
        bbox3d = det[6:12]
        x, y, x2, y2 = list(map(int, bbox))
        w = x2 - x
        h = y2 - y
        car_loc2d = (x + w // 2, y + h // 2)
        car_bbox2d = ((x - w // 2, y - h // 2), (x + w // 2, y + h // 2))
        left3d, right3d = bbox3d[:3], bbox3d[3:]
        car_loc3d = tuple(map(float, (left3d + right3d) / 2))
        assert len(car_loc3d) == 3
        car_bbox3d = (tuple(map(float, left3d)), tuple(map(float, right3d)))
        all_detections.append(obj_detection(DetectionId(frame_idx, i), car_loc3d, car_loc2d, car_bbox3d, car_bbox2d))
    # logger.info("all_detections", all_detections)
    all_detection_info = construct_all_detection_info(cam_polygon_mapping, ego_config, ego_trajectory, all_detections)
    return all_detection_info
