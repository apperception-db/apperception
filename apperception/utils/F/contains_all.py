from __future__ import annotations

import ast
from typing import TYPE_CHECKING, List

from .fake_fn import fake_fn

if TYPE_CHECKING:
    from ..fn_to_sql import GenSqlVisitor

ROAD_TYPES = {"road", "lane", "lanesection", "roadSection", "intersection"}


@fake_fn
def contains_all(visitor: "GenSqlVisitor", args: List[ast.expr]):
    arg_polygon, arg_points = args
    if not isinstance(arg_polygon, ast.Constant):
        raise Exception("Frist argument of contains_all should be a constant, recieved " + str(arg_polygon))

    polygon = visitor.visit(arg_polygon)[1:-1]
    points = visitor.visit(arg_points)

    if polygon not in ROAD_TYPES:
        raise Exception("polygon should be either " + " or ".join(ROAD_TYPES))

    size = None
    if isinstance(arg_points, ast.List):
        size = len(arg_points.elts)
    elif (isinstance(arg_points, ast.BinOp) and isinstance(arg_points.op, ast.MatMult) and isinstance(arg_points.left, ast.List)):
        size = len(arg_points.left.elts)

    return f"""(EXISTS(
        SELECT {polygon}.id
        FROM {polygon}
            JOIN SegmentPolygon
                ON SegmentPolygon.elementId = {polygon}.id
            JOIN unnest({points}) point
                ON ST_Covers(SegmentPolygon.elementPolygon, point)
        GROUP BY {polygon}.id
        HAVING COUNT(point) = {f'cardinality({points})' if size is None else size}
    ))"""
