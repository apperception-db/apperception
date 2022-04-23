from typing import Dict, List

import numpy as np
from pyquaternion import Quaternion

from apperception.data_types import Box, CameraConfig, TrackedObject


def recognize(camera_configs: List[CameraConfig], annotation):
    annotations: Dict[str, TrackedObject] = {}
    sample_token_to_time: Dict[str, int] = {}
    for config in camera_configs:
        if config.frame_id in sample_token_to_time:
            raise Exception("duplicate frame_id")
        sample_token_to_time[config.frame_id] = int(config.timestamp)

    for a in annotation.itertuples(index=False):
        sample_data_token = a.token_sample_data
        if sample_data_token not in sample_token_to_time:
            continue
        timestamp = sample_token_to_time[sample_data_token]
        item_id = a.instance_token
        if item_id not in annotations:
            annotations[item_id] = TrackedObject(a.category, [], [])

        box = Box(a.translation, a.size, Quaternion(a.rotation))

        corners = box.corners()
        bbox = np.transpose(corners[:, [3, 7]])  # type: ignore

        annotations[item_id].bboxes.append(bbox)
        annotations[item_id].timestamps.append(timestamp)
        annotations[item_id].itemHeading.append(a.heading)

    for item_id in annotations:
        timestamps = np.array(annotations[item_id].timestamps)
        bboxes = np.array(annotations[item_id].bboxes)
        itemHeadings = np.array(annotations[item_id].itemHeading)

        index = timestamps.argsort()

        annotations[item_id].timestamps = timestamps[index].tolist()
        annotations[item_id].bboxes = [bboxes[i, :, :] for i in index]
        annotations[item_id].itemHeading = itemHeadings[index].tolist()

    print("Recognization done, saving to database......")
    return annotations
