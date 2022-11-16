from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from .payload import Payload
    from .stages.stage import Stage


@dataclass
class Pipeline:
    stages: "List[Stage]" = field(default_factory=list)

    def __init__(self) -> None:
        self.stages = []

    def add_filter(self, filter: "Stage"):
        self.stages.append(filter)
        return self

    def run(self, payload: "Payload") -> "Payload":
        for filter in self.stages:
            payload = payload.filter(filter)
        return payload
