from __future__ import annotations

from typing import List

from apperception.predicate import GenSqlVisitor, PredicateNode, call_node


@call_node
def angle_excluding(visitor: "GenSqlVisitor", args: "List[PredicateNode]"):
    angle, angle_from, angle_to = args
    return f"angleExcluding({visitor.visit(angle)}, {visitor.visit(angle_from)}, {visitor.visit(angle_to)})"
