import math
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import numpy as np
from bitarray import bitarray

from .stage import Stage
from .tracking_3d.from_2d_and_depth import From2DAndDepth, Tracking3DResult
from .tracking_3d.tracking_3d import Tracking3D
from .utils.is_annotated import is_annotated

if TYPE_CHECKING:
    from ..payload import Payload


def facing_relative(prev_traj_point, next_traj_point, current_ego_heading):
    diff = (next_traj_point[0] - prev_traj_point[0], next_traj_point[1] - prev_traj_point[1])
    diff_heading = math.degrees(np.arctan2(diff[1], diff[0])) - 90
    result = ((diff_heading - current_ego_heading) % 360 + 360) % 360
    if result > 180:
        result -= 360
    return result


def facing_relative_check(obj_info, threshold, ego_config):
    facing_relative_filtered = {}
    for obj_id in obj_info:
        frame_idx = obj_info[obj_id]["frame_idx"].frame_idx
        trajectory = obj_info[obj_id]["trajectory"]
        ego_heading = ego_config.iloc[frame_idx.tolist()].egoHeading.tolist()
        filtered_idx = frame_idx[: len(ego_heading) - 1][
            [
                facing_relative(trajectory[i], trajectory[i + 1], ego_heading[i]) > threshold
                for i in range(len(ego_heading) - 1)
            ]
        ]
        facing_relative_filtered[obj_id] = filtered_idx
    return facing_relative_filtered


class FilterCarFacingSideway(Stage):
    def __call__(self, payload: "Payload") -> "Tuple[Optional[bitarray], Optional[list]]":
        if not is_annotated(Tracking3D, payload):
            # payload = payload.filter(From2DAndDepth())
            raise Exception()

        keep = bitarray(payload.keep)
        metadata = payload.metadata
        assert metadata is not None
        for i, (f, m) in enumerate(zip(payload.video, metadata)):
            if m is None:
                continue

            keep[i] = 0
            trackings_3d: "Dict[float, Tracking3DResult] | None" = From2DAndDepth.get(m)
            if trackings_3d is None:
                continue
            for t in trackings_3d.values():
                if t.prev is not None:
                    _from = t.prev
                else:
                    _from = t
                if t.next is not None:
                    _to = t.next
                else:
                    _to = t

                angle = facing_relative(_from.point, _to.point, f.ego_heading)
                if (50 < angle and angle < 135) or (-135 < angle and angle < -50):
                    keep[i] = 1
        return keep, None
