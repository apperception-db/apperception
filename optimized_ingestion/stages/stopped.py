from typing import TYPE_CHECKING, Dict, Optional, Tuple

from bitarray import bitarray

from apperception.database import database

from .stage import Stage

if TYPE_CHECKING:
    from ..payload import Payload


"""
Filters all frames where the ego is stopped, since this will mean that there is a red light and as
such, there won't be cars going in the intersection in the same direction (assuming traffic rules)
"""


class Stopped(Stage):
    def __init__(self, min_stopped_frames: int, stopped_threshold: float) -> None:
        super().__init__()
        self.min_stopped_frames = min_stopped_frames
        self.stopped_threshold = stopped_threshold

    def _run(self, payload: "Payload") -> "Tuple[Optional[bitarray], Optional[Dict[str, list]]]":
        keep = bitarray()
        frames = list(payload.video)
        stopped = set()
        for i in range(1, len(frames)):
            _frame = frames[i]
            _prev_frame = frames[i - 1]

            current_point = f"'POINT ({' '.join([*map(str, _frame.ego_translation)])})'"
            prev_point = f"'POINT ({' '.join([*map(str, _prev_frame.ego_translation)])})'"

            query = f"SELECT ABS(ST_Distance({current_point}, {prev_point}))"
            dist = database.execute(query)[0][0]

            query = f"SELECT minDistance({current_point}, 'intersection')"
            dist_intersection = database.execute(query)[0][0]

            if dist <= self.stopped_threshold and dist_intersection <= 5:
                # make sure that actually close to an intersection
                stopped.add(i)
                stopped.add(i - 1)

        for i in range(len(frames)):
            min_stopped = set()
            for j in range(i - self.min_stopped_frames, i + self.min_stopped_frames + 1):
                min_stopped.add(j)

            if min_stopped.issubset(stopped):
                keep.append(True)
            else:
                keep.append(False)

        return keep, None
