from __future__ import annotations

import ast
from typing import List

from apperception.predicate import GenSqlVisitor, PredicateNode, call_node

from .common import ROAD_TYPES


@call_node
def same_region(visitor: "GenSqlVisitor", args: "List[PredicateNode]"):
    type_, traj1, traj2 = args
    if not isinstance(type_, ast.Constant) or type_.value.lower() not in ROAD_TYPES:
        raise Exception(f"Unsupported road type: {type_}")

    return f"sameRegion({','.join(map(visitor, [type_, traj1, traj2]))})"
