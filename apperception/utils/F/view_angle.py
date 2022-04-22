from __future__ import annotations

import ast
from typing import TYPE_CHECKING, List

from .fake_fn import fake_fn

if TYPE_CHECKING:
    from ..fn_to_sql import GenSqlVisitor


@fake_fn
def view_angle(visitor: "GenSqlVisitor", args: List[ast.expr]):
    arg_obj_traj, arg_cam_heading, arg_cam_loc, arg_time = args
    
    if isinstance(arg_obj_traj, ast.Attribute):
        value = arg_obj_traj.value
        if not isinstance(value, ast.Name):
            raise Exception("First argument of view_angle should be trajectory")

        cont_point_attr = arg_obj_traj.attr
        if cont_point_attr == "traj" or cont_point_attr == "trajCentroids":
            object_positions = f"{visitor.eval_vars[value.id]}.trajCentroids"
        elif cont_point_attr == "bbox":
            raise Exception("We do not support bbox yet")
        else:
            raise Exception("First argument of contained should be trajectory")
    elif isinstance(arg_obj_traj, ast.Name):
        object_positions = f"{visitor.eval_vars[arg_obj_traj.id]}.trajCentroids"
    elif isinstance(arg_obj_traj, ast.Constant):
        object_positions = arg_obj_traj.value
    else:
        raise Exception("First argument of contained should be trajectory", str(arg_obj_traj))

    if isinstance(arg_cam_heading, ast.Attribute):
        value = arg_cam_heading.value
        if not isinstance(value, ast.Name):
            raise Exception("Problem with arg_geoms input contained function", str(arg_cam_heading))
        cont_point_attr = arg_cam_heading.attr
        cam_heading = f"{visitor.eval_vars[value.id]}.{cont_point_attr}"
    elif isinstance(arg_cam_heading, ast.Name):
        cam_heading = f"{visitor.eval_vars[arg_cam_heading.id]}"
    elif isinstance(arg_cam_heading, ast.Constant):
        cam_heading = arg_cam_heading.value
    else:
        raise Exception("Problem with arg_geoms input contained function", str(arg_cam_heading))

    if isinstance(arg_cam_loc, ast.Attribute):
        value = arg_cam_loc.value
        if not isinstance(value, ast.Name):
            raise Exception("Problem with arg_cam_loc input contained function", str(arg_cam_loc))
        cont_point_attr = arg_cam_loc.attr
        cam_loc = f"{visitor.eval_vars[value.id]}.{cont_point_attr}"
    elif isinstance(arg_cam_loc, ast.Name):
        cam_loc = f"{visitor.eval_vars[arg_cam_loc.id]}"
    elif isinstance(arg_cam_loc, ast.Constant):
        cam_loc = arg_cam_loc.value
    else:
        raise Exception("Problem with arg_cam_loc input contained function", str(arg_cam_loc))
    
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
