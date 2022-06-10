from __future__ import annotations

import ast
from typing import TYPE_CHECKING, List

from .fake_fn import fake_fn

if TYPE_CHECKING:
    from ..fn_to_sql import GenSqlVisitor


@fake_fn
def facing_relative(visitor: "GenSqlVisitor", args: List[ast.expr]):
    arg_heading1, arg_heading2, *arg_time = args

    if len(arg_time) == 0:
        return f"facingRelative({determine_heading(visitor, arg_heading1)}, {determine_heading(visitor, arg_heading2)})"
    return f"facingRelative({determine_heading(visitor, arg_heading1)}, {determine_heading(visitor, arg_heading2)}, {visitor.visit(arg_time[0])})"


def determine_heading(visitor: "GenSqlVisitor", arg: ast.expr):
    if isinstance(arg, ast.Attribute):
        value = arg.value
        attr = arg.attr
        if attr == "cam":
            heading = f"{visitor.visit(value)}.cameraHeading"
        elif attr == "ego":
            heading = f"{visitor.visit(value)}.egoHeading"
        else:
            heading = f"{visitor.visit(value)}.itemHeadings"
    elif isinstance(arg, ast.Name):
        heading = f"{visitor.visit(arg)}.itemHeadings"
    else:
        heading = f"{visitor.visit(arg)}"
    return heading
