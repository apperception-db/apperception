from __future__ import annotations

import ast
from typing import TYPE_CHECKING, List

from apperception.predicate import call_node, GenSqlVisitor, PredicateNode


@call_node
def like(visitor: "GenSqlVisitor", args: "List[PredicateNode]"):
    if len(args) != 2:
        raise Exception("like accepts 2 arguments")
    return " LIKE ".join(map(visitor.visit, args))
