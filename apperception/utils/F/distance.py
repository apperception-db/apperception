from __future__ import annotations

from typing import List

from apperception.predicate import GenSqlVisitor, PredicateNode, call_node


@call_node
def distance(visitor: "GenSqlVisitor", args: "List[PredicateNode]"):
    points = args[:2]
    if len(args) > 2:
        time = args[2]
        points = (p @ time for p in points)

    return f"distance({', '.join(map(visitor, points))})"
