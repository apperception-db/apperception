from typing import TYPE_CHECKING, Type


if TYPE_CHECKING:
    from ..filter import Filter
    from ...payload import Payload


def is_filtered(cls: "Type[Filter]", payload: "Payload"):
    if payload.metadata is None:
        return False

    for k, m in zip(payload.keep, payload.metadata):
        if k and cls.get(m) is None:
            return False

    return True
