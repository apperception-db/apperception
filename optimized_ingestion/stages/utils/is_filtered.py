from typing import TYPE_CHECKING, Type

if TYPE_CHECKING:
    from ...payload import Payload
    from ..stage import Stage


def is_filtered(cls: "Type[Stage]", payload: "Payload"):
    if payload.metadata is None:
        return False

    for k, m in zip(payload.keep, payload.metadata):
        if k and cls.get(m) is None:
            return False

    return True
