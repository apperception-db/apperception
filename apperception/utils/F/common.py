from apperception.predicate import CameraTableNode, ObjectTableNode, PredicateNode, TableAttrNode


HEADINGS = {
    "trajCentroids": "itemHeadings",
    "egoTranslation": "egoHeading",
    "cameraTranslation": "cameraHeading",
}

ROAD_TYPES = {"road", "lane", "lanesection", "roadSection", "intersection", "roadsection"}


def get_heading(arg: "PredicateNode"):
    if isinstance(arg, CameraTableNode):
        arg = arg.ego
    elif isinstance(arg, ObjectTableNode):
        arg = arg.traj

    if isinstance(arg, TableAttrNode) and arg.shorten:
        arg = getattr(arg.table, HEADINGS[arg.name])

    return arg