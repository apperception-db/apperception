from __future__ import annotations

import ast
from typing import List

from apperception.predicate import BinOpNode, CameraTableNode, GenSqlVisitor, ObjectTableNode, PredicateNode, TableAttrNode, call_node


HEADINGS = {
    "trajCentroids": "itemHeadings",
    "egoTranslation": "egoHeading",
    "cameraTranslation": "cameraHeading"
}


@call_node
def facing_relative(visitor: "GenSqlVisitor", args: "List[PredicateNode]"):
    arg_heading1, arg_heading2, *arg_time = args
    headings = arg_heading1, arg_heading2

    headings = map(get_heading, headings)
    if len(arg_time) == 0:
        headings = (h @ arg_time for h in headings)
    return f"facingRelative({', '.join(map(visitor.visit, headings))})"


def get_heading(arg: "PredicateNode"):
    if isinstance(arg, BinOpNode) and arg.op == 'matmul':
        return _get_heading(arg.left) @ arg.right
    return _get_heading(arg)


def _get_heading(arg: "PredicateNode"):
    if isinstance(arg, CameraTableNode):
        arg = arg.ego
    elif isinstance(arg, ObjectTableNode):
        arg = arg.traj

    if isinstance(arg, TableAttrNode) and arg.shorten == True:
        arg = getattr(arg.table, HEADINGS[arg.name])

    return arg

