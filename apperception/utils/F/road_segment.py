from __future__ import annotations

import ast
from typing import TYPE_CHECKING, List

from .fake_fn import fake_fn

if TYPE_CHECKING:
    from ..fn_to_sql import GenSqlVisitor

ROAD_TYPES = {"road", "lane", "lanesection", "roadsection", "intersection", "lanewithrightlane"}


@fake_fn
def road_segment(visitor: "GenSqlVisitor", args: List[ast.expr]):
    arg_type = args[0]
    if isinstance(arg_type, ast.Constant):
        if arg_type.value.lower() not in ROAD_TYPES:
            raise Exception(f"Unsupported road type: {arg_type}")
        else:
            return f"roadSegment('{arg_type.value}')"
    else:
        raise Exception(f"Unsupported road type: {arg_type}")
