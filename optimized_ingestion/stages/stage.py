from typing import Any, Dict, List, Optional, Tuple

from bitarray import bitarray

from ..payload import Payload


class Stage:
    def __call__(self, payload: "Payload") -> "Tuple[Optional[bitarray], Optional[list]]":
        return payload.keep, payload.metadata

    @classmethod
    def classname(cls):
        return ".".join(_get_classnames(cls))

    @classmethod
    def get(cls, d: "Dict[str, Any]"):
        classname = cls.classname()
        for k, v in [*d.items()][::-1]:
            if k.startswith(classname):
                return v
        return None


def _get_classnames(cls: "type") -> "List[str]":
    if cls == Stage:
        return []
    return [*_get_classnames(cls.__base__), cls.__name__]
