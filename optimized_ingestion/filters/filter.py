from typing import Optional, Tuple

from bitarray import bitarray
from optimized_ingestion.payload import Payload


class Filter:
    def __init__(self) -> None:
        pass

    def __call__(self, payload: "Payload") -> "Tuple[Optional[bitarray], Optional[list]]":
        return payload.keep, payload.metadata
