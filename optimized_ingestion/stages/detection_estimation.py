from ..payload import Payload
from .stage import Stage, StageOutput


class DetectionEstimation(Stage):
    def _run(self, payload: "Payload") -> "StageOutput":
        return super()._run(payload)