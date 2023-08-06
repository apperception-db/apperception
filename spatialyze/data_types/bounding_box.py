from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BoundingBox:
    x1: int
    y1: int
    x2: int
    y2: int

    def is_in(self, other: BoundingBox):
        if other.is_whole_frame():
            return True

        return (
            other.x1 <= self.x1
            and self.x2 <= other.x2
            and other.y1 <= self.y1
            and self.y2 <= other.y2
        )

    def to_tuples(self):
        return (self.y1, self.x1), (self.y2, self.x2)

    def is_whole_frame(self):
        return self.x1 == -1 and self.y1 == -1 and self.x2 == -1 and self.y2 == -1

    @property
    def area(self):
        if self.is_whole_frame():
            return float("inf")
        return (self.x2 - self.x1) * (self.y2 - self.y1)
