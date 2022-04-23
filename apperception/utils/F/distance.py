from __future__ import annotations

import ast
from typing import TYPE_CHECKING, List, Optional

from .fake_fn import fake_fn

if TYPE_CHECKING:
    from ..fn_to_sql import GenSqlVisitor

#   DISTANCE(cam.egoTranslation, t2.centroid, cam.timestamp) < 200


@fake_fn
def distance(visitor: "GenSqlVisitor", args: List[ast.expr]):
    arg_a, arg_b = args[0], args[1]
    arg_time: Optional[ast.expr] = None
    if len(args) > 2:
        arg_time = args[2]

    if isinstance(arg_a, ast.Attribute):
        value = arg_a.value
        attr = arg_a.attr
        if attr == "traj":
            a = f"{visitor.visit(value)}.trajCentroids"
        elif attr == "ego":
            a = f"{visitor.visit(value)}.egoTranslation"
        elif attr == "cam":
            a = f"{visitor.visit(value)}.camTranslation"
        else:
            raise Exception("we dont support other location yet")
    elif isinstance(arg_a, ast.Name):
        if arg_a.id == "road":
            a = visitor.visit(arg_a)
        else:
            a = f"{visitor.visit(arg_a)}.trajCentroids"
    else:
        a = visitor.visit(arg_a)

    if isinstance(arg_b, ast.Attribute):
        value = arg_b.value
        attr = arg_b.attr
        if attr == "traj":
            b = f"{visitor.visit(value)}.trajCentroids"
        elif attr == "ego":
            b = f"{visitor.visit(value)}.egoTranslation"
        elif attr == "cam":
            b = f"{visitor.visit(value)}.camTranslation"
        else:
            raise Exception("we dont support other location yet")
    elif isinstance(arg_b, ast.Name):
        if arg_b.id == "road":
            b = visitor.visit(arg_b)
        else:
            b = f"{visitor.visit(arg_b)}.trajCentroids"
    else:
        b = visitor.visit(arg_b)

    return f"distance({a}, {b}, {visitor.visit(arg_time)})" if arg_time else f"distance({a}, {b})"
