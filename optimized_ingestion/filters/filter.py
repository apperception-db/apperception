from typing import Any, Dict, List, Tuple
from frame import Frame

class Filter:
    def __init__(self) -> None:
        pass

    def filter(self, frames: List[Frame], metadata: Dict[Any, Any]) -> Tuple[List[Frame], Dict[Any, Any]]:
        return frames, metadata
