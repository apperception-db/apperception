from typing import Dict, Tuple

from ...payload import Payload
from ...types import DetectionId
from ..detection_2d.detection_2d import Detection2D
from ..detection_estimation import DetectionEstimation
from ..detection_estimation.detection_estimation import DetectionInfo
from ..detection_estimation.utils import trajectory_3d
from ..stage import Stage
from ..tracking_2d.strongsort import StrongSORT
from .construct_segment_trajectory import SegmentPoint, calibrate

SegmentTrajectoryMetadatum = Dict[int, SegmentPoint]


class SegmentTrajectory(Stage[SegmentTrajectoryMetadatum]):
    # @cache
    def _run(self, payload: "Payload"):
        if Detection2D.get(payload) is None:
            raise Exception()

        metadata: "list[dict[int, SegmentPoint]]" = [dict() for _ in range(len(payload.video))]
        obj_trajectories = construct_trajectory(payload)
        print('obj_trajectories', len(obj_trajectories))
        for oid, t in obj_trajectories.items():
            print(oid, len(t))
            trajectory_3d = [tt[0] for tt in t]
            detection_infos = [tt[1] for tt in t]
            frame_indices = [tt[2] for tt in t]

            calibrated_trajectory = calibrate(trajectory_3d, detection_infos, frame_indices, payload)
            for t in calibrated_trajectory:
                t.obj_id = oid

            calibrated_trajectory.sort(key=lambda t: t.timestamp)

            for before, after in zip(calibrated_trajectory[:-1], calibrated_trajectory[1:]):
                before.next = after
                after.prev = before

            for t in calibrated_trajectory:
                idx, _ = t.detection_id
                assert oid not in metadata[idx]
                metadata[idx][oid] = t

        return None, {self.classname(): metadata}


def construct_trajectory(source: "Payload"):
    obj_3d_trajectories: "dict[int, list[Tuple[trajectory_3d, DetectionInfo, int]]]" = {}
    trajectories = StrongSORT.get(source)
    assert trajectories is not None

    detection_infos = DetectionEstimation.get(source)
    assert detection_infos is not None

    detection_info_map: "dict[DetectionId, DetectionInfo]" = {}
    for d in detection_infos:
        for dd in d:
            detection_id = dd.detection_id
            assert detection_id not in detection_info_map
            detection_info_map[detection_id] = dd

    for frame_idx, frame in enumerate(trajectories):
        for obj_id, obj_trajectory in frame.items():
            if obj_id not in obj_3d_trajectories:
                obj_3d_trajectories[obj_id] = []

            detection_id = obj_trajectory.detection_id
            assert detection_id in detection_info_map, (detection_id, [*detection_info_map.keys()])
            detection_info = detection_info_map[detection_id]
            obj_3d_trajectories[obj_id].append((
                trajectory_3d(detection_info.car_loc3d, detection_info.timestamp),
                detection_info,
                frame_idx,
            ))

    return obj_3d_trajectories