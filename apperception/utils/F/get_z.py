from __future__ import annotations

import ast
from typing import TYPE_CHECKING, List, Optional

from .fake_fn import fake_fn

if TYPE_CHECKING:
    from ..fn_to_sql import GenSqlVisitor


@fake_fn
def get_z(visitor: "GenSqlVisitor", args: List[ast.expr]):
    arg_location = args[0]
    arg_time: Optional[ast.expr] = None
    if len(args) > 1:
        arg_time = args[1]

    if isinstance(arg_location, ast.Attribute):
        value = arg_location.value
        attr = arg_location.attr
        if attr == "traj":
            location = f"{visitor.visit(value)}.trajCentroids"
        elif attr == "trans":
            location = f"{visitor.visit(value)}.translations"
        elif attr == "ego":
            location = f"{visitor.visit(value)}.egoTranslation"
        elif attr == "cam":
            location = f"{visitor.visit(value)}.cameraTranslation"
        else:
            raise Exception("we dont support other location yet")
    elif isinstance(arg_location, ast.Name):
        if arg_location.id == "road":
            location = visitor.visit(arg_location)
        else:
            location = f"{visitor.visit(arg_location)}.trajCentroids"
    else:
        location = visitor.visit(arg_location)
    return f"getZ({location}, {visitor.visit(arg_time)})" if arg_time else f"getZ({location})"
