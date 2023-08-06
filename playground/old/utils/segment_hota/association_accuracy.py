from typing import Callable

import numpy as np
import numpy.typing as npt

from scipy.optimize import linear_sum_assignment

from .object_type import object_type  # type: ignore

from .types import SegmentDetection


def _create_trajectories(detections: "list[SegmentDetection]"):
    trajectories: "dict[str, list[SegmentDetection]]" = {}

    for detection in detections:
        key = detection.oid
        if key not in trajectories:
            trajectories[key] = []
        trajectories[key].append(detection)
    
    for trajectory in trajectories.values():
        trajectory.sort(key = lambda t: t.timestamp)

    return trajectories


def relaxed_detections_matched(prediction: "SegmentDetection", groundtruth: "SegmentDetection"):
    conditions = (
        prediction.segment == groundtruth.segment,
        # prediction.segment.polygon == groundtruth.segment.polygon,
        # prediction.type == groundtruth.type
    )

    return all(conditions)


def _get_association_iou(
    pd_trajectory: "list[SegmentDetection]",
    gt_trajectory: "list[SegmentDetection]",
    detections_matched: "Callable[[SegmentDetection, SegmentDetection], bool]",
):
    pd_types = set(d.type for d in pd_trajectory)
    # assert len(pd_types) == 1, pd_types
    if len(pd_types) > 1 and 'car' in pd_types:
        pd_type = 'car'
    else:
        pd_type = [*pd_types][0]

    gt_types = set(d.type for d in gt_trajectory)
    assert len(gt_types) == 1, gt_types
    gt_type = [*gt_types][0]

    # if gt_type != pd_type:
    #     print(gt_type, pd_type)
    #     return 0

    tp, fp, fn = 0, 0, 0
    frame_to_pd = dict()
    frame_to_gt = dict()
    for pd in pd_trajectory:
        key = pd.fid
        assert key not in frame_to_pd
        frame_to_pd[key] = pd
    for gt in gt_trajectory:
        key = gt.fid
        assert key not in frame_to_gt
        frame_to_gt[key] = gt
    
    for key, gt in frame_to_gt.items():
        if key not in frame_to_pd:
            fp += 1
        else:
            if detections_matched(frame_to_pd[key], gt):
                tp += 1
            else:
                fp += 1
                fn += 1
    
    for key in frame_to_pd:
        if key not in frame_to_gt:
            fn += 1


    # i, j = 0, 0
    # while i < len(pd_trajectory) and j < len(gt_trajectory):
    #     pd: "SegmentDetection" = pd_trajectory[i]
    #     gt: "SegmentDetection" = gt_trajectory[j]

    #     if pd.timestamp == gt.timestamp:
    #         if detections_matched(pd, gt):
    #             tp += 1
    #         else:
    #             fp += 1
    #             fn += 1
    #         i += 1
    #         j += 1
    #         pass
    #     elif pd.timestamp < gt.timestamp:
    #         fp += 1
    #         i += 1
    #     elif pd.timestamp > gt.timestamp:
    #         fn += 1
    #         j += 1
    #     else:
    #         raise Exception()
    
    # fp += len(pd_trajectory) - i
    # fn += len(gt_trajectory) - j

    # print(tp, fp, fn)

    if tp + fp + fn == 0:
        return 0

    return tp / (tp + fp + fn)


def _match_detections(
    predictions: "list[SegmentDetection]",
    groundtruths: "list[SegmentDetection]",
    simplified_loc_ious: "dict[str, dict[str, float]]",
):
    n, m = map(len, [groundtruths, predictions])
    iou = np.zeros((n, m))

    for i, gt in enumerate(groundtruths):
        for j, pd in enumerate(predictions):
            iou[i, j] = simplified_loc_ious.get(gt.oid, {}).get(pd.oid, 0)
            if pd.timestamp == gt.timestamp:
                iou[i, j] = simplified_loc_ious[gt.oid][pd.oid]
    
    res: "tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]" = linear_sum_assignment(iou, maximize=True)
    row_ind, col_ind = res

    to_prediction: "dict[str, str]" = {}
    to_groundtruth: "dict[str, str]" = {}
    for r, c in zip(map(int, row_ind), map(int, col_ind)):
        gt_did = groundtruths[r].did
        pd_did = predictions[c].did

        to_prediction[gt_did] = pd_did
        to_groundtruth[pd_did] = gt_did

    return to_prediction, to_groundtruth


def _get_all_association_ious(
    pd_trajectories: "dict[str, list[SegmentDetection]]",
    gt_trajectories: "dict[str, list[SegmentDetection]]",
    detections_matched: "Callable[[SegmentDetection, SegmentDetection], bool]",
):
    ious: "dict[str, dict[str, float]]" = {}
    for gt_oid, gt_trajectory in gt_trajectories.items():
        assert gt_oid not in ious, [*ious.keys()]
        ious[gt_oid] = gt_iou = {}
        for pd_oid, pd_trajectory in pd_trajectories.items():
            assert pd_oid not in gt_iou, [*gt_iou.keys()]
            gt_iou[pd_oid] = _get_association_iou(pd_trajectory, gt_trajectory, detections_matched)
            # if gt_iou[pd_oid] != 0:
            #     print(pd_oid, gt_oid, gt_iou[pd_oid])
    return ious


def _match_tracks(
    pd_trajectories: "dict[str, list[SegmentDetection]]",
    gt_trajectories: "dict[str, list[SegmentDetection]]",
    association_ious: "dict[str, dict[str, float]]"
) -> "tuple[dict[str, str], dict[str, str]]":
    n, m = map(len, [gt_trajectories, pd_trajectories])
    iou = np.zeros((n, m))

    gt_oids = [*gt_trajectories]
    pd_oids = [*pd_trajectories]
    for i, gt_oid in enumerate(gt_oids):
        for j, pd_oid in enumerate(pd_oids):
            iou[i, j] = association_ious.get(gt_oid, {}).get(pd_oid, 0)
    res: "tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]" = linear_sum_assignment(iou, maximize=True)
    row_ind, col_ind = res

    to_prediction: "dict[str, str]" = {}
    to_groundtruth: "dict[str, str]" = {}
    for r, c in zip(map(int, row_ind), map(int, col_ind)):
        gt_did = gt_oids[r]
        pd_did = pd_oids[c]

        to_prediction[gt_did] = pd_did
        to_groundtruth[pd_did] = gt_did

    return to_prediction, to_groundtruth


def association_accuracy(
    predictions: "list[SegmentDetection]",
    groundtruths: "list[SegmentDetection]",
    alpha: "float",
):
    pd_trajectories = _create_trajectories(predictions)
    gt_trajectories = _create_trajectories(groundtruths)

    # Note: 2 detections are matched only if they are at the same timestamp.

    # Calculate simplified association IOU
    # For any two tracks, IOU is number of segment detections they share at the same timestamps.
    # TODO: explain why not just use this score?
    simplified_association_ious = _get_all_association_ious(pd_trajectories, gt_trajectories, relaxed_detections_matched)

    # Match segment detection points using the simplified association IOU as its cost for the hungarian matching algorithm
    # The idea is that the detection points pair that their tracks share more segment detections are more likely to be the same detection
    to_prediction, _ = _match_detections(predictions, groundtruths, simplified_association_ious)
    # print(to_prediction)

    def detections_matched(prediction: "SegmentDetection", groundtruth: "SegmentDetection"):
        # return prediction.type == groundtruth.type and to_prediction[groundtruth.did] == prediction.did
        if groundtruth.did not in to_prediction:
            return False
        return to_prediction[groundtruth.did] == prediction.did

    # Calculate association IOU
    # Only allow the pair of detections that has been matched to be counted as matching detections when find association IOU
    # This is not the same as the simplified version:
    #   - simplified association IOU:  any 2 detections that are in the same segment are matched.
    #   -       this association IOU:  2 detections need to be matched from from the _match_detection.
    association_ious = _get_all_association_ious(pd_trajectories, gt_trajectories, detections_matched)

    association_to_pd, _ = _match_tracks(pd_trajectories, gt_trajectories, association_ious)
    # print(association_to_pd)

    tp = 0
    sum_association_iou: "float" = 0
    # print('association ious -----------------------------------------------------------------')
    # print("\n".join([str((k, v)) for k, v in association_ious.items() if any(vv != 0 for vv in v.values())]))
    # print('-----------------------------------------------------------------')
    # print(len(association_to_pd))
    for gt_oid, pd_oid in association_to_pd.items():
        # print(gt_oid, pd_oid)
        iou = association_ious[gt_oid][pd_oid]
        sum_association_iou += iou
        if iou != 0:
            tp += 1
    
    if tp == 0:
        return 0
    
    return sum_association_iou / tp
