from __future__ import annotations

from typing import TYPE_CHECKING, List

from apperception.predicate import ObjectTableNode, PredicateNode, TableAttrNode, call_node, GenSqlVisitor


HEADINGS = {
    "trajCentroids": "itemHeadings",
    "egoTranslation": "egoHeading",
    "cameraTranslation": "cameraHeading"
}


@call_node
def ahead(visitor: "GenSqlVisitor", args: "List[PredicateNode]"):
    arg_obj1, arg_obj2, arg_time = args

    loc1 = visitor.visit(arg_obj1)

    if isinstance(arg_obj2, ObjectTableNode):
        arg_obj2 = arg_obj2.traj

    if not isinstance(arg_obj2, TableAttrNode):
        raise Exception("we dont support other location yet")

    attr = arg_obj2.name
    if attr not in HEADINGS:
        raise Exception("we dont support other location yet")

    loc2 = visitor.visit(arg_obj2)
    heading = visitor.visit(getattr(arg_obj2.table, HEADINGS[attr]))

    return f"ahead({loc1}, {loc2}, {heading}, {visitor.visit(arg_time)})"
