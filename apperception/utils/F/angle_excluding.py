from __future__ import annotations

from typing import List

from apperception.predicate import GenSqlVisitor, PredicateNode, call_node


@call_node
def angle_excluding(visitor: "GenSqlVisitor", args: "List[PredicateNode]"):
    return f"angleExcluding({','.join(map(visitor, args))})"
