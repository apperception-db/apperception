from __future__ import annotations

import ast
from typing import TYPE_CHECKING, List

from .fake_fn import fake_fn

if TYPE_CHECKING:
    from ..fn_to_sql import GenSqlVisitor


@fake_fn
def road_direction(visitor: "GenSqlVisitor", args: List[ast.expr]):
    arg_location = args[0]
    arg_time = ""
    if len(args) > 1:
        arg_time = args[1]
    
    if isinstance(arg_location, ast.Attribute):
        value = arg_location.value
        attr = arg_location.attr
        if attr == "traj":
            location = f"{visitor.eval_vars[value.id]}.trajCentroids"
        elif attr == "ego":
            location = f"{visitor.eval_vars[value.id]}.egoTranslation"
        elif attr == "cam":
            location = f"{visitor.eval_vars[value.id]}.camTranslation"
        else:
            "we dont support other location yet"
    elif isinstance(arg_location, ast.Name):
        if arg_location.id == "road":
            location = f"{visitor.visit(arg_location.id)}"
        else:
            location = f"{visitor.eval_vars[arg_location.id]}.trajCentroids"
    else:
        location = f"{visitor.eval_vars[arg_location]}"
    return f"roadDirection({location}, {visitor.visit(arg_time)})" if arg_time else f"roadDirection({location})"

    