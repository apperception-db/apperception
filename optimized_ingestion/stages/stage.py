import time
from typing import TYPE_CHECKING, Any, Generic, Type, TypeVar

from bitarray import bitarray

if TYPE_CHECKING:
    from ..payload import Payload


T = TypeVar('T')


class Stage(Generic[T]):
    benchmark: "list[dict]"
    keeps: "list[tuple[int, int]]"

    def __new__(cls, *_, **__):
        obj = super(Stage, cls).__new__(cls)
        obj.benchmark = []
        obj.keeps = []
        return obj

    def _run(self, payload: "Payload") -> "tuple[bitarray | None, dict[str, list[T]] | None]":
        return payload.keep, payload.metadata

    def run(self, payload: "Payload") -> "tuple[bitarray | None, dict[str, list[T]] | None]":
        keep_before = payload.keep
        s = time.time()
        out = self._run(payload)
        e = time.time()
        keep_after = out[0]

        if keep_after is None:
            keep_after = keep_before
        _keep = keep_after & keep_before

        self.benchmark.append({
            "name": payload.video.videofile,
            "runtime": e - s,
            "keep": (sum(_keep), sum(keep_before))
        })

        return out

    @classmethod
    def classname(cls):
        return ".".join(_get_classnames(cls))

    _T = TypeVar('_T')

    @classmethod
    def get(cls: "Type[Stage[_T]]", d: "dict[str, list] | Payload") -> "list[_T] | None":
        if not isinstance(d, dict):
            d = d.metadata

        classname = cls.classname()
        for k, v in reversed(d.items()):
            if k.startswith(classname):
                return v
        return None
    
    @classmethod
    def encode_json(cls, o: "Any") -> "Any":
        return None


def _get_classnames(cls: "type") -> "list[str]":
    if cls == Stage:
        return []
    return [*_get_classnames(cls.__base__), cls.__name__]
