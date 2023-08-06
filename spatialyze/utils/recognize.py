from typing import Dict, List

import numpy as np
from pyquaternion import Quaternion

from spatialyze.data_types import Box, CameraConfig, TrackedObject


def recognize(camera_configs: List[CameraConfig], annotation):
    annotations: Dict[str, TrackedObject] = {}
    sample_token_to_time: Dict[str, int] = {}
    for config in camera_configs:
        if config.frame_id in sample_token_to_time:
            raise Exception("duplicate frame_id")
        sample_token_to_time[config.frame_id] = int(config.timestamp)

    for a in annotation.itertuples(index=False):
        sample_data_tokens = [sdt for sdt in a.sample_data_tokens if sdt in sample_token_to_time]
        for sample_data_token in sample_data_tokens:
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
            annotations[item_id].translations.append(a.translation)

    for item_id in annotations:
        timestamps = np.array(annotations[item_id].timestamps)
        bboxes = np.array(annotations[item_id].bboxes)
        itemHeadings = np.array(annotations[item_id].itemHeading)
        translations = np.array(annotations[item_id].translations)

        index = timestamps.argsort()

        annotations[item_id].timestamps = timestamps[index].tolist()
        annotations[item_id].bboxes = [bboxes[i, :, :] for i in index]
        annotations[item_id].itemHeading = itemHeadings[index].tolist()
        annotations[item_id].translations = translations[index].tolist()

    print("Recognization done, saving to database......")
    return annotations
