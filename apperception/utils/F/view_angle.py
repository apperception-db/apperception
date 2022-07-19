from __future__ import annotations

from typing import List

from apperception.predicate import BinOpNode, CameraTableNode, GenSqlVisitor, ObjectTableNode, PredicateNode, TableAttrNode, call_node
from .common import get_heading


@call_node
def view_angle(visitor: "GenSqlVisitor", args: "List[PredicateNode]"):
    object, pov = args[:2]

    if len(args) == 3:
        object = object @ args[2]

    if isinstance(pov, CameraTableNode):
        pov = pov.cam

    if isinstance(pov, TableAttrNode):
        heading = get_heading(pov)
    elif isinstance(pov, BinOpNode) and pov.op == 'matmul':
        left = pov.left
        if isinstance(left, ObjectTableNode):
            left = left.traj
        pov = left @ pov.right
        heading = get_heading(left)

    return f"viewAngle({','.map(visitor, [object, heading, pov])})"
