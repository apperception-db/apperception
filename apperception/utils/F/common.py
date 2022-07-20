from apperception.predicate import (BinOpNode, CastNode, PredicateNode,
                                    TableAttrNode)

HEADINGS = {
    "trajCentroids": "itemHeadings",
    "egoTranslation": "egoHeading",
    "cameraTranslation": "cameraHeading",
}

ROAD_TYPES = {
    "road",
    "lane",
    "lanesection",
    "roadSection",
    "intersection",
    "roadsection",
    "lanewithrightlane",
}


def get_heading(arg: "PredicateNode"):
    if isinstance(arg, TableAttrNode) and arg.shorten:
        arg = getattr(arg.table, HEADINGS[arg.name])

    return arg


def get_heading_at_time(arg: "PredicateNode"):
    if isinstance(arg, BinOpNode) and arg.op == "matmul":
        return CastNode("real", get_heading_at_time(arg.left) @ arg.right)
    return get_heading(arg)
