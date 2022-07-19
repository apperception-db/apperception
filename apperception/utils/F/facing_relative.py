from __future__ import annotations

from typing import List

from apperception.predicate import (GenSqlVisitor, PredicateNode,
                                    call_node)

from .common import get_heading_at_time


@call_node
def facing_relative(visitor: "GenSqlVisitor", args: "List[PredicateNode]"):
    args = map(get_heading_at_time, args)
    return f"facingRelative({', '.join(map(visitor, args))})"
