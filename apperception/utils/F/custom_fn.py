from __future__ import annotations
from typing import List
from apperception.predicate import GenSqlVisitor, PredicateNode, call_node


def custom_fn(name: str):
    @call_node
    def fn(visitor: "GenSqlVisitor", args: "List[PredicateNode]"):
        return f"{name}({', '.join(map(visitor.visit, args))})"
    
    return fn
