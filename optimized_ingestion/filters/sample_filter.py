from .filter import Filter
from typing import Any, Dict, List, Tuple
from frame import Frame

class SampleFilter(Filter):
    sampling_rate: int

    def __init__(self, sampling_rate) -> None:
        self.sampling_rate = sampling_rate

    def filter(self, frames: List[Frame], metadata: Dict[Any, Any]) -> Tuple[List[Frame], Dict[Any, Any]]:
        return frames[::self.sampling_rate], metadata