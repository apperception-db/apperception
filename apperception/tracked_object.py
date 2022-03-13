from dataclasses import dataclass, field
from typing import List, Union

import numpy as np


@dataclass
class TrackedObject:
    object_type: str
    bboxes: Union[List[np.ndarray], np.ndarray] = field(default_factory=list)
    frame_num: List[int] = field(default_factory=list)

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, TrackedObject)
            and self.object_type == other.object_type
            and self.frame_num == other.frame_num
            and np.array_equal(np.array(self.bboxes), np.array(other.bboxes))
        )

    def equal(self, other) -> bool:
        if not isinstance(other, TrackedObject) or self.object_type != other.object_type:
            return False

        s_frame_num = np.array(self.frame_num)
        o_frame_num = np.array(other.frame_num)

        s_indices = s_frame_num.argsort()
        o_indices = o_frame_num.argsort()

        return np.array_equal(s_frame_num[s_indices], o_frame_num[o_indices]) and np.array_equal(
            np.array(self.bboxes)[s_indices], np.array(other.bboxes)[o_indices]
        )
