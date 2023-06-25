from apperception.predicate import GenSqlVisitor, PredicateNode, call_node

ROAD_TYPES = {"road", "lane", "lanesection", "roadsection", "intersection", "lanegroup"}


@call_node
def is_other_roadtype(visitor: "GenSqlVisitor", args: "list[PredicateNode]"):
    raise Exception('Should not be used')
    # (roadtype,) = args
    # assert isinstance(roadtype, LiteralNode)

    # roadtype = roadtype.value.lower()
    # assert roadtype in ROAD_TYPES
    # assert roadtype != 'road'

    # return " OR ".join(
    #     f"SegmentPolygon.__RoadType__{t}__"
    #     for t in OTHER_ROAD_TYPES[roadtype]
    # )
