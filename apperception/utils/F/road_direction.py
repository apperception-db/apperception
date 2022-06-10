from __future__ import annotations

import ast
from typing import TYPE_CHECKING, List

from .fake_fn import fake_fn

if TYPE_CHECKING:
    from ..fn_to_sql import GenSqlVisitor


@fake_fn
def road_direction(visitor: "GenSqlVisitor", args: List[ast.expr]):
    arg_location, arg_default = args[0], args[-1]
    arg_time = None
    if len(args) > 2:
        arg_time = args[1]

    if isinstance(arg_location, ast.Attribute):
        value = arg_location.value
        attr = arg_location.attr
        if attr == "traj":
            location = f"{visitor.visit(value)}.trajCentroids"
        elif attr == "ego":
            location = f"{visitor.visit(value)}.egoTranslation"
        elif attr == "cam":
            location = f"{visitor.visit(value)}.cameraTranslation"
        else:
            raise Exception("we dont support other location yet")
    elif isinstance(arg_location, ast.Name):
        if arg_location.id == "road":
            location = f"{visitor.visit(arg_location)}"
        else:
            location = f"{visitor.visit(arg_location)}.trajCentroids"
    else:
        location = f"{visitor.visit(arg_location)}"
    return (
        f"roadDirection({location}, {visitor.visit(arg_time)}, {determine_heading(visitor, arg_default)})"
        if arg_time
        else f"roadDirection({location}, {determine_heading(visitor, arg_default)})"
    )


def determine_heading(visitor: "GenSqlVisitor", arg: ast.expr):
    if isinstance(arg, ast.Attribute):
        value = arg.value
        attr = arg.attr
        if attr == "cam":
            heading = f"{visitor.visit(value)}.cameraHeading"
        elif attr == "ego":
            heading = f"{visitor.visit(value)}.egoHeading"
        else:
            # TODO: Need to make a time parameter for roadDirection() in order for this work
            heading = f"{visitor.visit(value)}.itemHeadings"
    elif isinstance(arg, ast.Name):
        heading = f"{visitor.visit(arg)}.itemHeadings"
    else:
        heading = f"{visitor.visit(arg)}"
    return heading
