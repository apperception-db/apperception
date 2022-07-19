from __future__ import annotations

from typing import List

from apperception.predicate import (GenSqlVisitor, ObjectTableNode,
                                    PredicateNode, TableAttrNode, call_node)
from .common import get_heading


@call_node
def ahead(visitor: "GenSqlVisitor", args: "List[PredicateNode]"):
    obj1, obj2, time = args

    if isinstance(obj2, ObjectTableNode):
        obj2 = obj2.traj

    if not isinstance(obj2, TableAttrNode):
        raise Exception("we dont support other location yet")

    heading = get_heading(obj2)
    return f"ahead({','.join(visitor(p @ time) for p in [obj1, obj2, heading])})"
