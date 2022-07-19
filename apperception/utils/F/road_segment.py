from __future__ import annotations

import ast
from typing import TYPE_CHECKING, List
from apperception.predicate import LiteralNode, call_node, GenSqlVisitor, PredicateNode


ROAD_TYPES = {"road", "lane", "lanesection", "roadsection", "intersection", "lanewithrightlane"}


@call_node
def road_segment(visitor: "GenSqlVisitor", args: "List[PredicateNode]"):
    arg_type = args[0]
    if not isinstance(arg_type, LiteralNode) or not isinstance(arg_type.value, str) or arg_type.value.lower() not in ROAD_TYPES:
        raise Exception(f"Unsupported road type: {arg_type}")
    return f"roadSegment('{arg_type.value}')"
