from __future__ import annotations

from typing import List

from apperception.predicate import CameraTableNode, GenSqlVisitor, PredicateNode, call_node
from .common import HEADINGS, get_heading

@call_node
def convert_camera(visitor: "GenSqlVisitor", args: "List[PredicateNode]"):
    object, camera = args[:2]

    if len(args) == 3:
        object = object @ args[2]

    if isinstance(camera, CameraTableNode):
        camera = camera.cam
    heading = get_heading(camera)

    return f"ConvertCamera({','.join(map(visitor, [object, camera, heading]))})"
