from typing import Optional, Tuple

from bitarray import bitarray

from ..payload import Payload
from .stage import Stage


class Sample(Stage):
    sampling_rate: int

    def __init__(self, sampling_rate):
        self.sampling_rate = sampling_rate

    def __call__(self, payload: "Payload") -> "Tuple[Optional[bitarray], Optional[list]]":
        keep = bitarray(len(payload.keep))
        keep[:: self.sampling_rate] = 1
        return keep, None
