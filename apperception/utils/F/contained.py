from __future__ import annotations

from typing import List

from apperception.predicate import GenSqlVisitor, PredicateNode, call_node


@call_node
def contained(visitor: "GenSqlVisitor", args: "List[PredicateNode]"):
    if len(args) == 2:
        point, geoms = args
        time = None
    elif len(args) == 3:
        point, geoms, time = args
    else:
        raise Exception("contained accept either 2 or 3 arguments")

    if time is not None:
        point = point @ time

    return f"contained({','.join(map(visitor, [point, geoms]))})"
