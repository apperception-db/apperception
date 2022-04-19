from __future__ import annotations
import ast
from typing import TYPE_CHECKING, Tuple

from .fake_fn import fake_fn

if TYPE_CHECKING:
    from ..predicate import PredicateVisitor


@fake_fn
def convert_camera(visitor: "PredicateVisitor", args: Tuple[ast.AST, ast.AST, ast.AST]):
    [_from, cam] = visitor.tables
    [geometry_type, camera_type, timestamp] = args
    geometry_table = f"{_from}.trajCentroid" if geometry_type == "obj.traj" else "general_bbox.bbox"
    camera_config = "cameras.ego_translation" if camera_type == "ego_cam" else "cameras.camera_translation"
    return f"ConvertCamera({geometry_table}, {camera_config}, cameras.timestamp)"
