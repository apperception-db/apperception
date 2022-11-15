from __future__ import annotations

from typing import List

from apperception.predicate import GenSqlVisitor, LiteralNode, PredicateNode, call_node

from .common import ROAD_TYPES


@call_node
def road_segment(visitor: "GenSqlVisitor", args: "List[PredicateNode]"):
    table = args[0]
    if (
        not isinstance(table, LiteralNode)
        or not isinstance(table.value, str)
        or table.value not in ROAD_TYPES
    ):
        raise Exception(f"Unsupported road type: {table}")
    return f"roadSegment({visitor(table)})"
