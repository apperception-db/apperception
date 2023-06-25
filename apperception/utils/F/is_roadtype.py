from apperception.predicate import GenSqlVisitor, PredicateNode, call_node

ROAD_TYPES = {"road", "lane", "lanesection", "roadsection", "intersection"}


@call_node
def is_roadtype(visitor: "GenSqlVisitor", args: "list[PredicateNode]"):
    raise Exception('Should not be used')
    # (roadtype,) = args

    # assert isinstance(roadtype, LiteralNode)
    # assert roadtype.value.lower() in ROAD_TYPES

    # return f"SegmentPolygon.__RoadType__{roadtype.value}__"
