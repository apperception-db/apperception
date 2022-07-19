from __future__ import annotations

import ast
from typing import TYPE_CHECKING, List

from apperception.predicate import CameraTableNode, GenSqlVisitor, ObjectTableNode, PredicateNode, call_node

from .common import get_heading

@call_node
def road_direction(visitor: "GenSqlVisitor", args: "List[PredicateNode]"):
    location = args[0]
    heading = get_heading(location)
    if len(args) > 1:
        if isinstance(heading.table, ObjectTableNode):
            location = location @ args[1]
            heading = heading @ args[1]
    
    return f"roadDirection({','.join(visitor, [location, heading])})"
