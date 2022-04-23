from typing import Tuple
from apperception.utils.F.fake_fn import fake_fn


def test_fake_fn():
    @fake_fn
    def test_fn1(_, args: Tuple[int, bool, str]):
        a, b, c = args
        return f"{a} {b} {c}"

    assert test_fn1() is None
    assert test_fn1.fn(None, (3, True, "test"))
