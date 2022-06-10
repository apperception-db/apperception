from __future__ import annotations

import ast
from typing import TYPE_CHECKING, List

from .fake_fn import fake_fn

if TYPE_CHECKING:
    from ..fn_to_sql import GenSqlVisitor


@fake_fn
def angle_between(visitor: "GenSqlVisitor", args: List[ast.expr]):
    angle, angle_from, angle_to = args
    return f"angleBetween({visitor.visit(angle)}, {visitor.visit(angle_from)}, {visitor.visit(angle_to)})"
