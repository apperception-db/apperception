from typing import TYPE_CHECKING, Type

if TYPE_CHECKING:
    from ...payload import Payload
    from ..stage import Stage


def is_annotated(cls: "Type[Stage]", payload: "Payload"):
    metadata = cls.get(payload.metadata)

    if metadata is None:
        return False

    for k, m in zip(payload.keep, metadata):
        if k and m is None:
            return False
    return True
