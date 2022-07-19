from __future__ import annotations

from typing import List

from apperception.predicate import (ArrayNode, BinOpNode, GenSqlVisitor,
                                    LiteralNode, PredicateNode, call_node)

ROAD_TYPES = {"road", "lane", "lanesection", "roadSection", "intersection"}


@call_node
def contains_all(visitor: "GenSqlVisitor", args: "List[PredicateNode]"):
    arg_polygon, arg_points = args
    if not isinstance(arg_polygon, LiteralNode):
        raise Exception(
            "Frist argument of contains_all should be a constant, recieved " + str(arg_polygon)
        )

    polygon = visitor.visit(arg_polygon)[1:-1]
    points = visitor.visit(arg_points)

    if polygon not in ROAD_TYPES:
        raise Exception("polygon should be either " + " or ".join(ROAD_TYPES))

    size = None
    if isinstance(arg_points, ArrayNode):
        size = len(arg_points.exprs)
    elif (
        isinstance(arg_points, BinOpNode)
        and arg_points.op == "matmul"
        and isinstance(arg_points.left, ArrayNode)
    ):
        size = len(arg_points.left.exprs)

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
