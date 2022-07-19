from __future__ import annotations

from typing import List

from apperception.predicate import (BinOpNode, CameraTableNode, GenSqlVisitor,
                                    ObjectTableNode, PredicateNode,
                                    TableAttrNode, call_node)

from .common import get_heading_at_time


@call_node
def view_angle(visitor: "GenSqlVisitor", args: "List[PredicateNode]"):
    object, pov = args

    heading = get_heading_at_time(pov)
    return f"viewAngle({','.join(map(visitor, [object, heading, pov]))})"
