from __future__ import annotations

import ast
from typing import TYPE_CHECKING, List

from .fake_fn import fake_fn

if TYPE_CHECKING:
    from ..fn_to_sql import GenSqlVisitor


@fake_fn
def convert_camera(visitor: "GenSqlVisitor", args: List[ast.expr]):
    arg_object, arg_camera = args

    if isinstance(arg_object, ast.Attribute):
        value = arg_object.value
        if not isinstance(value, ast.Name):
            raise Exception("First argument of convert_camera should be trajectory")

        camera_attr = arg_object.attr
        if camera_attr == "traj":
            object_positions = f"{visitor.eval_vars[value.id]}.trajCentroid"
        else:
            raise Exception("First argument of convert_camera should be trajectory")
    elif isinstance(arg_object, ast.Name):
        raise Exception("First argument of convert_camera should be trajectory")
    else:
        raise Exception("First argument of convert_camera should be trajectory", str(arg_object))

    if isinstance(arg_camera, ast.Attribute):
        value = arg_camera.value
        if not isinstance(value, ast.Name):
            raise Exception()
        name = value.id
        if arg_camera.attr != "ego":
            raise Exception("Second argument of convert_camera should be camera or its ego car")
        camera_attr = arg_camera.attr
    elif isinstance(arg_camera, ast.Name):
        name = arg_camera.id
        camera_attr = "camera_translation"
    else:
        raise Exception("Second argument of convert_camera should be camera or its ego car")

    return f"ConvertCamera({object_positions}, {visitor.eval_vars[name]}.{camera_attr}, {visitor.eval_vars[name]}.timestamp)"
