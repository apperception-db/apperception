from typing import TYPE_CHECKING, Any, Callable, Generic, Tuple, TypeVar

if TYPE_CHECKING:
    from ..predicate import PredicateVisitor


T1 = TypeVar('T1')
class FakeFn(Generic[T1]):  # noqa: E302
    _fn: Tuple[Callable[["PredicateVisitor", T1], str]]

    def __init__(self, fn: Callable[["PredicateVisitor", T1], str]):
        self._fn = (fn,)

    def __call__(self, *_: T1) -> Any:
        return None

    @property
    def fn(self):
        return self._fn[0]


T2 = TypeVar('T2')
def fake_fn(fn: Callable[["PredicateVisitor", T2], str]) -> FakeFn[T2]:  # noqa: E302
    return FakeFn(fn)
