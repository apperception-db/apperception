import time
from bitarray import bitarray
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from ..payload import Payload


StageOutput = Tuple[Optional[bitarray], Optional[Dict[str, list]]]


class Stage:
    runtimes: "List[float]"

    def __new__(cls):
        obj = object.__new__(cls)
        obj.runtimes = []
        return obj

    def _run(self, payload: "Payload") -> "StageOutput":
        return payload.keep, payload.metadata

    def run(self, payload: "Payload") -> "StageOutput":
        s = time.time()
        out = self._run(payload)
        e = time.time()

        self.runtimes.append(e - s)

        return out

    @classmethod
    def classname(cls):
        return ".".join(_get_classnames(cls))

    @classmethod
    def get(cls, d: "Dict[str, list] | Payload"):
        if not isinstance(d, dict):
            d = d.metadata

        classname = cls.classname()
        for k, v in reversed(d.items()):
            if k.startswith(classname):
                return v
        return None


def _get_classnames(cls: "type") -> "List[str]":
    if cls == Stage:
        return []
    return [*_get_classnames(cls.__base__), cls.__name__]
