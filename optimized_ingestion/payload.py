from dataclasses import dataclass
from typing import List, TYPE_CHECKING

from bitarray import bitarray

if TYPE_CHECKING:
    from frame import Frame


@dataclass
class Payload:
    frames: "List[Frame]"
    keep: "bitarray"
