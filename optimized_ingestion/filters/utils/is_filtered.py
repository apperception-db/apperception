from typing import TYPE_CHECKING, Type

if TYPE_CHECKING:
    from ...payload import Payload
    from ..filter import Filter


def is_filtered(cls: "Type[Filter]", payload: "Payload"):
    if payload.metadata is None:
        return False

    for k, m in zip(payload.keep, payload.metadata):
        if k and cls.get(m) is None:
            return False

    return True
