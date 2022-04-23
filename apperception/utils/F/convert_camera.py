from __future__ import annotations

import ast
from typing import TYPE_CHECKING, List

from .fake_fn import fake_fn

if TYPE_CHECKING:
    from ..fn_to_sql import GenSqlVisitor


@fake_fn
def convert_camera(visitor: "GenSqlVisitor", args: List[ast.expr]):
    arg_object, arg_camera, arg_time = args

    if isinstance(arg_object, ast.Attribute):
        value = arg_object.value
        if not isinstance(value, ast.Name):
            raise Exception("First argument of convert_camera should be trajectory")

        camera_attr = arg_object.attr
        if camera_attr == "traj":
            object_positions = f"{visitor.eval_vars[value.id]}.trajCentroids"
        elif camera_attr == "bbox":
            raise Exception("We do not support bbox yet")
        else:
            raise Exception("First argument of convert_camera should be trajectory")
    elif isinstance(arg_object, ast.Name):
        object_positions = f"{visitor.eval_vars[arg_object.id]}.trajCentroids"
    else:
        raise Exception("First argument of convert_camera should be trajectory", str(arg_object))

    if isinstance(arg_camera, ast.Attribute):
        value = arg_camera.value
        if not isinstance(value, ast.Name):
            raise Exception()
        name = value.id
        if arg_camera.attr != "ego":
            raise Exception("Second argument of convert_camera should be camera or its ego car")
        camera_attr = "egoTranslation"
    elif isinstance(arg_camera, ast.Name):
        name = arg_camera.id
        camera_attr = "cameraTranslation"
    else:
        raise Exception("Second argument of convert_camera should be camera or its ego car")

    if isinstance(arg_camera, ast.Attribute):
        value = arg_camera.value
        if not isinstance(value, ast.Name):
            raise Exception()
        name = value.id
        if arg_camera.attr != "ego":
            raise Exception("Second argument of convert_camera should be camera or its ego car")
        camera_heading_attr = "egoHeading"
    elif isinstance(arg_camera, ast.Name):
        name = arg_camera.id
        camera_heading_attr = "cameraHeading"
    else:
        raise Exception("Second argument of convert_camera should be camera or its ego car")

    

    return f"ConvertCamera({object_positions}, {visitor.eval_vars[name]}.{camera_attr}, {visitor.eval_vars[name]}.{camera_heading_attr}, {visitor.visit(arg_time)})"
