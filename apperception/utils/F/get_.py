from __future__ import annotations

from typing import List, Literal

from apperception.predicate import GenSqlVisitor, PredicateNode, call_node


def get_(axis: Literal["x", "y", "z"]):
    @call_node
    def get_(visitor: "GenSqlVisitor", args: "List[PredicateNode]"):
        location = args[0]

        if len(args) > 1:
            location = location @ args[1]

        return f"st_{axis}({visitor(location)})"

    return get_
