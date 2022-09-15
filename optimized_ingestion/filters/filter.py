from typing import TYPE_CHECKING, Optional, Tuple

from bitarray import bitarray

if TYPE_CHECKING:
    from payload import Payload


class Filter:
    def __init__(self) -> None:
        pass

    def __call__(self, payload: "Payload") -> "Tuple[Optional[bitarray], Optional[list]]":
        return payload.keep, payload.metadata
