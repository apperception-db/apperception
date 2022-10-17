from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from bitarray import bitarray

if TYPE_CHECKING:
    from ..payload import Payload


StageOutput = Tuple[Optional[bitarray], Optional[Dict[str, list]]]


class Stage:
    def __call__(self, payload: "Payload") -> "StageOutput":
        return payload.keep, payload.metadata

    @classmethod
    def classname(cls):
        return ".".join(_get_classnames(cls))

    @classmethod
    def get(cls, d: "Dict[str, list]"):
        classname = cls.classname()
        for k, v in reversed(d.items()):
            if k.startswith(classname):
                return v
        return None


def _get_classnames(cls: "type") -> "List[str]":
    if cls == Stage:
        return []
    return [*_get_classnames(cls.__base__), cls.__name__]
