from typing import List

from apperception.predicate import GenSqlVisitor, PredicateNode, call_node

from .common import get_heading_at_time


@call_node
def convert_camera(visitor: "GenSqlVisitor", args: "List[PredicateNode]"):
    object, camera = args[:2]
    heading = get_heading_at_time(camera)
    return f"ConvertCamera({','.join(map(visitor, [object, camera, heading]))})"
