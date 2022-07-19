from __future__ import annotations

from typing import List

from apperception.predicate import GenSqlVisitor, PredicateNode, call_node


@call_node
def distance(visitor: "GenSqlVisitor", args: "List[PredicateNode]"):
    arg_ab = args[:2]
    if len(args) > 2:
        time = args[2]
        arg_ab = (a @ time for a in arg_ab)

    return f"distance({', '.join(map(visitor.visit, arg_ab))})"
