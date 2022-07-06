from __future__ import annotations

import ast
from typing import TYPE_CHECKING, List

from .fake_fn import fake_fn

if TYPE_CHECKING:
    from ..fn_to_sql import GenSqlVisitor


@fake_fn
def view_angle(visitor: "GenSqlVisitor", args: List[ast.expr]):
    arg_obj, arg_point_of_view, arg_time = args

    if isinstance(arg_obj, ast.Attribute):
        value = arg_obj.value
        if not isinstance(value, ast.Name):
            raise Exception("obj not supported yet")
        cont_point_attr = arg_obj.attr
        if cont_point_attr == "traj" or cont_point_attr == "trajCentroids":
            object_positions = f"{visitor.eval_vars[value.id]}.trajCentroids"
        elif cont_point_attr == "trans":
            object_positions = f"{visitor.eval_vars[value.id]}.translations"
        elif cont_point_attr == "bbox":
            raise Exception("We do not support bbox yet")
        else:
            raise Exception("First argument of contained should be geometry type")
    elif isinstance(arg_obj, ast.Name):
        object_positions = f"{visitor.eval_vars[arg_obj.id]}.trajCentroids"
    elif isinstance(arg_obj, ast.Constant):
        object_positions = arg_obj.value
    else:
        raise Exception("First argument of contained should be geometry type", str(arg_obj))

    if isinstance(arg_point_of_view, ast.Attribute):
        value = arg_point_of_view.value
        view_point_attr = arg_point_of_view.attr
        if not isinstance(value, ast.Name):
            raise Exception("point of view not supported yet")
        if view_point_attr == "ego":
            cam_loc = f"{visitor.eval_vars[value.id]}.egoTranslation"
            cam_heading = f"{visitor.eval_vars[value.id]}.egoHeading"
        elif view_point_attr == "camera":
            cam_loc = f"{visitor.eval_vars[value.id]}.cameraTranslation"
            cam_heading = f"{visitor.eval_vars[value.id]}.cameraHeading"
        elif view_point_attr == "cameraAbs" or view_point_attr == "cameraAbsolute":
            cam_loc = f"{visitor.eval_vars[value.id]}.cameraTranslationAbs"
            cam_heading = f"{visitor.eval_vars[value.id]}.cameraHeading"
        else:
            raise Exception("only support camera attribute")
    elif isinstance(arg_point_of_view, ast.Name):
        cam_heading = f"{visitor.eval_vars[arg_point_of_view.id]}.itemHeadings"
        cam_loc = f"{visitor.eval_vars[arg_point_of_view.id]}.trajCentroids"
    elif isinstance(arg_point_of_view, ast.Constant):
        cam_heading = arg_point_of_view.value
        cam_loc = arg_point_of_view.value
    else:
        raise Exception("Problem with arg_geoms input contained function", str(arg_point_of_view))

    if isinstance(arg_time, ast.Attribute):
        value = arg_time.value
        if not isinstance(value, ast.Name):
            raise Exception("Problem with arg_time input contained function", str(arg_time))
        cont_point_attr = arg_time.attr
        timet = f"{visitor.eval_vars[value.id]}.{cont_point_attr}"
    elif isinstance(arg_time, ast.Name):
        timet = f"{visitor.eval_vars[arg_time.id]}"
    elif isinstance(arg_time, ast.Constant):
        timet = arg_time.value
    else:
        raise Exception("Problem with arg_time input contained function", str(arg_time))

    return f"viewAngle({object_positions}, {cam_heading}, {cam_loc}, {timet})"
