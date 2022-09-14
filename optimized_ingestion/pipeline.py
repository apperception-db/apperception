
from typing import List
from filters.filter import Filter
from frame import Frame
from queue import Queue


class Pipeline:
    filters: "Queue[Filter]" = []

    def __init__(self) -> None:
        self.filters = Queue()

    def add_filter(self, filter: Filter) -> None:
        self.filters.put(filter)

    def run(self, frames: List[Frame]) -> List[Frame]:
        metadata = {}
        while not self.filters.empty():
            current_filter = self.filters.get()
            frames, metadata = current_filter.filter(frames, metadata)
        return frames
