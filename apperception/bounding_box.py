from __future__ import annotations
from dataclasses import dataclass


@dataclass
class BoundingBox:
    x1: int
    y1: int
    x2: int
    y2: int

    def is_in(self, other: BoundingBox):
        if other.x1 == -1 and other.y1 == -1 and other.x2 == -1 and other.y2 == -1:
            return True

        return other.x1 <= self.x1 and self.x2 <= other.x2 and other.y1 <= self.y1 and self.y2 <= other.y2

    def to_tuples(self):
        return ((self.x1, self.y1), (self.x2, self.y2))


WHOLE_FRAME = BoundingBox(-1, -1, -1, -1)