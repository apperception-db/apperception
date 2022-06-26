from __future__ import annotations

import ast
from typing import TYPE_CHECKING, List, Optional

from .fake_fn import fake_fn

if TYPE_CHECKING:
    from ..fn_to_sql import GenSqlVisitor

ROAD_TYPES = {"road", "lane", "lanesection", "roadSection", "intersection", "roadsection"}


@fake_fn
def same_region(visitor: "GenSqlVisitor", args: List[ast.expr]):
    arg_type, arg_traj1, arg_traj2 = args[0], args[1], args[2]
    arg_time: Optional[ast.expr] = None
    if len(args) > 3:
        arg_time = args[3]

    if not isinstance(arg_type, ast.Constant) or arg_type.value.lower() not in ROAD_TYPES:
        raise Exception(f"Unsupported road type: {arg_type}")

    if isinstance(arg_traj1, ast.Attribute):
        value = arg_traj1.value
        attr = arg_traj1.attr
        if attr == "traj":
            traj1 = f"{visitor.visit(value)}.trajCentroids"
        elif attr == "ego":
            traj1 = f"{visitor.visit(value)}.egoTranslation"
        elif attr == "cam":
            traj1 = f"{visitor.visit(value)}.cameraTranslation"
        else:
            raise Exception("we dont support other location yet")
    elif isinstance(arg_traj1, ast.Name):
        if arg_traj1.id == "road":
            traj1 = visitor.visit(arg_traj1)
        else:
            traj1 = f"{visitor.visit(arg_traj1)}.trajCentroids"
    else:
        traj1 = visitor.visit(arg_traj1)

    if isinstance(arg_traj2, ast.Attribute):
        value = arg_traj2.value
        attr = arg_traj2.attr
        if attr == "traj":
            traj2 = f"{visitor.visit(value)}.trajCentroids"
        elif attr == "ego":
            traj2 = f"{visitor.visit(value)}.egoTranslation"
        elif attr == "cam":
            traj2 = f"{visitor.visit(value)}.cameraTranslation"
        else:
            raise Exception("we dont support other location yet")
    elif isinstance(arg_traj2, ast.Name):
        if arg_traj2.id == "road":
            traj2 = visitor.visit(arg_traj2)
        else:
            traj2 = f"{visitor.visit(arg_traj2)}.trajCentroids"
    else:
        traj2 = visitor.visit(arg_traj2)

    if arg_time:
        return f"sameRegion('{arg_type.value}', {traj1}, {traj2}, {visitor.visit(arg_time)})"
    else:
        return f"sameRegion('{arg_type.value}', {traj1}, {traj2})"
