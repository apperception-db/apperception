from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from ...payload import Payload


def is_filtered(cls: type, payload: "Payload"):
    if payload.metadata is None:
        return False
    
    for k, m in zip(payload.keep, payload.metadata):
        if k and cls.__name__ not in m:
            return False

    return True