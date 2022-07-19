from __future__ import annotations

from typing import List

from apperception.predicate import (BinOpNode, GenSqlVisitor, PredicateNode,
                                    call_node)

from .common import get_heading


@call_node
def facing_relative(visitor: "GenSqlVisitor", args: "List[PredicateNode]"):
    headings = args[:2]

    headings = map(_get_heading, headings)
    if len(args) == 3:
        headings = (h @ args[2] for h in headings)
    return f"facingRelative({', '.join(map(visitor, headings))})"


def _get_heading(arg: "PredicateNode"):
    if isinstance(arg, BinOpNode) and arg.op == "matmul":
        return get_heading(arg.left) @ arg.right
    return get_heading(arg)
