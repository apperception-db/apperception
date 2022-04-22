from __future__ import annotations

import ast
from typing import TYPE_CHECKING, List

from .fake_fn import fake_fn

if TYPE_CHECKING:
    from ..fn_to_sql import GenSqlVisitor


@fake_fn
def facing_relative(visitor: "GenSqlVisitor", args: List[ast.expr]):
    arg_heading1, arg_heading2, arg_time = args

    return f"facingRelative({determine_heading(visitor, arg_heading1)}, {determine_heading(visitor, arg_heading2)}, {visitor.eval_vars[arg_time.value.id]}.timestamp)"


def determine_heading(visitor: "GenSqlVisitor", arg: ast.expr):
    if isinstance(arg, ast.Attribute):
        value = arg.value
        attr = arg.attr
        if attr == "cam":
            heading = f"{visitor.eval_vars[value.id]}.camHeading"
        elif attr == "ego":
            heading = f"{visitor.eval_vars[value.id]}.egoHeading"
        else:
            heading = f"{visitor.eval_vars[value.id]}.itemHeadings"
    elif isinstance(arg, ast.Name):
        heading = f"{visitor.eval_vars[arg.id]}.itemheadings"
    else:
        heading = f"{visitor.eval_vars[arg]}"
    return heading
