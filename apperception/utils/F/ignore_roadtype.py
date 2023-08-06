from apperception.predicate import GenSqlVisitor, PredicateNode, call_node

ROAD_TYPES = {"road", "lane", "lanesection", "roadsection", "intersection"}


@call_node
def ignore_roadtype(visitor: "GenSqlVisitor", args: "list[PredicateNode]"):
    return "ignore_roadtype()"
    # raise Exception('Snould not be used')
    # assert len(args) == 0, len(args)
    # return " OR ".join(f"SegmentPolygon.__RoadType__{t}__" for t in ROAD_TYPES)
