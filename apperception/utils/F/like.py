from typing import List

from apperception.predicate import GenSqlVisitor, PredicateNode, call_node


@call_node
def like(visitor: "GenSqlVisitor", args: "List[PredicateNode]") -> str:
    if len(args) != 2:
        raise Exception("like accepts 2 arguments")
    return " LIKE ".join(map(visitor, args))
