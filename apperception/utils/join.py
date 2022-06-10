from typing import Iterable


def join(array: Iterable, delim: str = ","):
    return delim.join(map(str, array))
