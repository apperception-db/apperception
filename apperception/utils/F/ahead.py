from __future__ import annotations

import ast
from typing import TYPE_CHECKING, List, Optional

from .fake_fn import fake_fn

if TYPE_CHECKING:
    from ..fn_to_sql import GenSqlVisitor

#   ahead(obj1, cam.ego, cam.timestamp)


@fake_fn
def ahead(visitor: "GenSqlVisitor", args: List[ast.expr]):
    arg_obj1, arg_obj2, arg_time = args

    if isinstance(arg_obj1, ast.Attribute):
        value = arg_obj1.value
        attr = arg_obj1.attr
        if attr == "traj":
            loc1 = f"{visitor.visit(value)}.trajCentroids"
        elif attr == "ego":
            loc1 = f"{visitor.visit(value)}.egoTranslation"
        elif attr == "cam":
            loc1 = f"{visitor.visit(value)}.cameraTranslation"
        else:
            raise Exception("we dont support other location yet")
    elif isinstance(arg_obj1, ast.Name):
        loc1 = f"{visitor.visit(arg_obj1)}.trajCentroids"
    else:
        loc1 = visitor.visit(arg_obj1)

    if isinstance(arg_obj2, ast.Attribute):
        value = arg_obj2.value
        attr = arg_obj2.attr
        if attr == "traj":
            loc2 = f"{visitor.visit(value)}.trajCentroids"
            heading = f"{visitor.visit(value)}.itemHeadings"
        elif attr == "ego":
            loc2 = f"{visitor.visit(value)}.egoTranslation"
            heading = f"{visitor.visit(value)}.egoHeading" 
        elif attr == "cam":
            loc2 = f"{visitor.visit(value)}.cameraTranslation"
            heading = f"{visitor.visit(value)}.cameraHeading"
        else:
            raise Exception("we dont support other location yet")
    elif isinstance(arg_obj2, ast.Name):
        loc2 = f"{visitor.visit(arg_obj2)}.trajCentroids"
        heading = f"{visitor.visit(arg_obj2)}.itemHeadings"
    else:
        raise Exception("we dont support other location yet") 

    return f"ahead({loc1}, {loc2}, {heading}, {visitor.visit(arg_time)})"
