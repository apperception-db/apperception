import time
from bitarray import bitarray
from typing import TYPE_CHECKING, Dict, Generic, List, Tuple, Type, TypeVar

if TYPE_CHECKING:
    from ..payload import Payload


T = TypeVar('T')


class Stage(Generic[T]):
    runtimes: "List[dict]"

    def __new__(cls, *_, **__):
        obj = super(Stage, cls).__new__(cls)
        obj.runtimes = []
        return obj

    def _run(self, payload: "Payload") -> "Tuple[bitarray | None, Dict[str, List[T]] | None]":
        return payload.keep, payload.metadata

    def run(self, payload: "Payload") -> "Tuple[bitarray | None, Dict[str, List[T]] | None]":
        s = time.time()
        out = self._run(payload)
        e = time.time()

        self.runtimes.append({
            "name": payload.video.videofile,
            "runtime": e - s
        })

        return out

    @classmethod
    def classname(cls):
        return ".".join(_get_classnames(cls))

    _T = TypeVar('_T')

    @classmethod
    def get(cls: "Type[Stage[_T]]", d: "Dict[str, list] | Payload") -> "List[_T] | None":
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
