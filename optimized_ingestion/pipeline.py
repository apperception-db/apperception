from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from .payload import Payload
    from .stages.stage import Stage


@dataclass
class Pipeline:
    filters: "List[Stage]" = field(default_factory=list)

    def __init__(self) -> None:
        self.filters = []

    def add_filter(self, filter: "Stage"):
        self.filters.append(filter)
        return self

    def run(self, payload: "Payload") -> "Payload":
        for filter in self.filters:
            payload = payload.filter(filter)
        return payload
