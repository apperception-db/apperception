from typing import Literal

from .custom_fn import custom_fn


def get_(axis: Literal["x", "y", "z"]):
    return custom_fn(f"st_{axis}", 1)
