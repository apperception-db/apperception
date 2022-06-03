from __future__ import annotations

import ast
from typing import TYPE_CHECKING, List

from .fake_fn import fake_fn

if TYPE_CHECKING:
    from ..fn_to_sql import GenSqlVisitor


@fake_fn
def like(visitor: "GenSqlVisitor", args: List[ast.expr]):
    if len(args) != 2:
        raise Exception("like accepts 2 arguments")
    arg_1, arg_2 = args
    return f"({visitor.visit(arg_1)} LIKE {visitor.visit(arg_2)})"
