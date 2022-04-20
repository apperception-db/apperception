from typing import TYPE_CHECKING, Any, Callable, Generic, Tuple, TypeVar

if TYPE_CHECKING:
    from ..fn_to_sql import GenSqlVisitor


CT = TypeVar("CT")
FT = TypeVar("FT")


class FakeFn(Generic[CT]):
    _fn: Tuple[Callable[["GenSqlVisitor", CT], str]]

    def __init__(self, fn: Callable[["GenSqlVisitor", CT], str]):
        self._fn = (fn,)

    def __call__(self, *_: CT) -> Any:
        return None

    @property
    def fn(self):
        return self._fn[0]


def fake_fn(fn: Callable[["GenSqlVisitor", FT], str]) -> FakeFn[FT]:
    return FakeFn(fn)
