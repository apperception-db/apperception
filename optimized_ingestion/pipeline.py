from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .payload import Payload
    from .stages.stage import Stage


@dataclass
class Pipeline:
    stages: "list[Stage]" = field(default_factory=list)

    def __init__(self) -> None:
        self.stages = []

    def add_filter(self, filter: "Stage") -> "Pipeline":
        self.stages.append(filter)
        return self

    def run(self, payload: "Payload") -> "Payload":
        for stage in self.stages:
            payload = payload.filter(stage)
        return payload
