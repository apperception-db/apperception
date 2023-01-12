import itertools
import multiprocessing
from typing import List, Tuple

from bitarray import bitarray
import torch
from tqdm import tqdm

from ...payload import Payload
from ..detection_2d.yolo_detection import YoloDetection
from . import construct_estimated_all_detection_info, generate_sample_plan_once
from .segment_mapping import map_imgsegment_roadsegment

from .utils import trajectory_3d
from ...utils.partition_by_cpus import partition_by_cpus
from ...video import Video


def _estimation(
    args: "Tuple[Video, int, List[trajectory_3d], List[torch.Tensor], int, int]"
) -> "list[int]":
    video, start_frame_num, ego_trajectory, dets, start, end = args

    if start_frame_num >= end:
        return [start_frame_num for _ in range(start, end)]

    out: "list[int]" = []
    if start_frame_num > start:
        out = [start_frame_num for _ in range(start, start_frame_num)]
        start = start_frame_num

    next_frame_num = start
    for i in range(start, end):
        current_ego_config = video[i]
        next_frame_num = i + 1
        cam_segment_mapping = map_imgsegment_roadsegment(current_ego_config)
        det = dets[i]
        all_detection_info = construct_estimated_all_detection_info(
            det,
            cam_segment_mapping,
            current_ego_config,
            ego_trajectory,
            start,
        )
        next_sample_plan, _ = generate_sample_plan_once(
            video,
            current_ego_config,
            cam_segment_mapping,
            next_frame_num,
            all_detection_info=all_detection_info
        )
        next_frame_num = next_sample_plan.get_next_frame_num(next_frame_num)
        out.append(next_frame_num)

    return out


def parallel_estimation(
    payload: "Payload",
    start_frame_num: "int",
    ego_trajectory: "List[trajectory_3d]",
) -> "Tuple[bitarray, None]":
    n_cpus = multiprocessing.cpu_count()
    frame_slices = partition_by_cpus(n_cpus, len(payload.video))

    dets = YoloDetection.get(payload)
    assert dets is not None

    with multiprocessing.Pool(n_cpus) as pool:
        inputs = [
            (payload.video, start_frame_num, ego_trajectory, [d.cpu() for d, _ in dets[start:end]], start, end)
            for start, end in frame_slices
        ]
        out = [*tqdm(pool.imap(_estimation, inputs), total=len(inputs))]

    skipped_frame_num = []
    next_frame_num: "int" = start_frame_num
    for i, _next_frame_num in enumerate(itertools.chain(*out)):
        if i < next_frame_num:
            skipped_frame_num.append(i)
            continue
        next_frame_num = _next_frame_num

    keep = bitarray(len(payload.video))
    keep[:] = 1
    for f in skipped_frame_num:
        keep[f] = 0

    return keep, None