import datetime
from typing import Tuple

import shapely
import shapely.geometry

from .object_type import object_type

from .types import SegmentDetection, Polygon, Segment


def _create_detections_map(detections: "list[SegmentDetection]"):
    detections_map: "dict[Tuple[Segment, str, str], list[SegmentDetection]]" = {}
    for det in detections:
        key = det.segment, det.fid, det.type
        if key not in detections_map:
            detections_map[key] = []
        detections_map[key].append(det)
    return detections_map


def detection_accuracy(
    predictions: "list[SegmentDetection]",
    groundtruths: "list[SegmentDetection]"
):
    # print(set([pd.type for pd in predictions]))
    # print(set([pd.type for pd in groundtruths]))
    predictions_map = _create_detections_map(predictions)
    groundtruths_map = _create_detections_map(groundtruths)
    
    tp: "int" = 0
    fp: "int" = 0
    fn: "int" = 0

    for key, pd_segment_detections in predictions_map.items():
        if key not in groundtruths_map:
            fp += len(pd_segment_detections)
        else:
            gt_segment_detections = groundtruths_map[key]
            tp += min(len(pd_segment_detections), len(gt_segment_detections))
            fp += max(0, len(pd_segment_detections) - len(gt_segment_detections))
            fn += max(0, len(gt_segment_detections) - len(pd_segment_detections))
    
    for key, gt_segment_detections in groundtruths_map.items():
        if key not in predictions_map:
            fn += len(gt_segment_detections)
    
    if tp + fp + fn == 0:
        return 0.

    print(tp, fp, fn)
    return tp / (tp + fp + fn)
