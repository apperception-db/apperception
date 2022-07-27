from __future__ import annotations

from typing import List

from apperception.predicate import (ArrayNode, BinOpNode, GenSqlVisitor,
                                    LiteralNode, PredicateNode, call_node)

ROAD_TYPES = {"road", "lane", "lanesection", "roadSection", "intersection"}


@call_node
def contains_all(visitor: "GenSqlVisitor", args: "List[PredicateNode]"):
    polygon, points = args
    if not isinstance(polygon, LiteralNode):
        raise Exception(
            "Frist argument of contains_all should be a constant, recieved " + str(polygon)
        )

    polygon_ = visitor(polygon)[1:-1]
    points_ = visitor(points)

    if polygon_ not in ROAD_TYPES:
        raise Exception("polygon should be either " + " or ".join(ROAD_TYPES))

    size = None
    if isinstance(points, ArrayNode):
        size = len(points.exprs)
    elif (
        isinstance(points, BinOpNode)
        and points.op == "matmul"
        and isinstance(points.left, ArrayNode)
    ):
        size = len(points.left.exprs)

    return f"""(EXISTS(
        SELECT {polygon_}.id
        FROM {polygon_}
            JOIN SegmentPolygon
                ON SegmentPolygon.elementId = {polygon_}.id
            JOIN unnest({points_}) point
                ON ST_Covers(SegmentPolygon.elementPolygon, point)
        GROUP BY {polygon_}.id
        HAVING COUNT(point) = {f'cardinality({points_})' if size is None else size}
    ))"""
