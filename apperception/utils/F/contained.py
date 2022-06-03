from __future__ import annotations

import ast
from typing import TYPE_CHECKING, List

from .fake_fn import fake_fn

if TYPE_CHECKING:
    from ..fn_to_sql import GenSqlVisitor


@fake_fn
def contained(visitor: "GenSqlVisitor", args: List[ast.expr]):
    if len(args) == 2:
        arg_cont_point, arg_geoms = args
        arg_time = None
    if len(args) == 3:
        arg_cont_point, arg_geoms, arg_time = args

    if isinstance(arg_cont_point, ast.Attribute):
        value = arg_cont_point.value
        if not isinstance(value, ast.Name):
            raise Exception("First argument of contained should be trajectory")

        cont_point_attr = arg_cont_point.attr
        if cont_point_attr == "traj" or cont_point_attr == "trajCentroids":
            object_positions = f"{visitor.eval_vars[value.id]}.trajCentroids"
        elif cont_point_attr == "ego":
            object_positions = f"{visitor.eval_vars[value.id]}.egoTranslation"
        elif cont_point_attr == "bbox":
            raise Exception("We do not support bbox yet")
        else:
            raise Exception("First argument of contained should be trajectory")
    elif isinstance(arg_cont_point, ast.Name):
        object_positions = f"{visitor.eval_vars[arg_cont_point.id]}.trajCentroids"
    elif isinstance(arg_cont_point, ast.Constant):
        object_positions = arg_cont_point.value
    else:
        raise Exception("First argument of contained should be trajectory", str(arg_cont_point))

    if isinstance(arg_geoms, ast.Attribute):
        value = arg_geoms.value
        if not isinstance(value, ast.Name):
            raise Exception("Problem with arg_geoms input contained function", str(arg_geoms))
        cont_point_attr = arg_geoms.attr
        geoms = f"{visitor.eval_vars[value.id]}.{cont_point_attr}"
    elif isinstance(arg_geoms, ast.Name):
        geoms = f"{visitor.eval_vars[arg_geoms.id]}"
    elif isinstance(arg_geoms, ast.Constant):
        geoms = arg_geoms.value
    else:
        geoms = f"{visitor.visit(arg_geoms)}"

    if arg_time is not None:
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

    if arg_time is not None:
        return f"contained({object_positions}, {geoms}, {timet})"
    else:
        return f"contained({object_positions}, {geoms})"
