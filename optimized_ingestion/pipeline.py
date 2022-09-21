from dataclasses import dataclass, field
from queue import Queue
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .payload import Payload
    from .stages.stage import Stage


@dataclass
class Pipeline:
    filters: "Queue[Stage]" = field(default_factory=Queue)

    def __init__(self) -> None:
        self.filters = Queue()

    def add_filter(self, filter: "Stage"):
        self.filters.put(filter)
        return self

    def run(self, payload: "Payload") -> "Payload":
        while not self.filters.empty():
            current_filter = self.filters.get()
            payload = payload.filter(current_filter)
        return payload
