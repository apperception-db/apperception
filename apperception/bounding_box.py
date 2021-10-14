from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BoundingBox:
    x0: int
    x1: int
    y0: int
    y1: int

    def covers(self, other: BoundingBox):
        return (
            self.x0 <= other.x0
            and other.x1 <= self.x1
            and self.y0 <= other.y0
            and other.y1 <= self.y1
        )
