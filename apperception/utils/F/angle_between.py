from __future__ import annotations

from typing import TYPE_CHECKING, List

from apperception.predicate import call_node

if TYPE_CHECKING:
    from apperception.predicate import GenSqlVisitor, PredicateNode


@call_node
def angle_between(visitor: "GenSqlVisitor", args: "List[PredicateNode]"):
    angle, angle_from, angle_to = args
    return f"angleBetween({visitor.visit(angle)}, {visitor.visit(angle_from)}, {visitor.visit(angle_to)})"
