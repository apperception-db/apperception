from typing import Any, Dict, Optional, Tuple, Type, TypeVar

from bitarray import bitarray

from ..payload import Payload


class Filter:
    def __init__(self) -> None:
        pass

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


def _get_classnames(cls: "type"):
    _name = []
    if cls.__base__ != object:
        _name = _get_classnames(cls.__base__)
    return _name + [cls.__name__]