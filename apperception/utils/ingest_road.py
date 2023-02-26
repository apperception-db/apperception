import json
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from apperception.database import Database


CREATE_POLYGON_SQL = """
CREATE TABLE IF NOT EXISTS SegmentPolygon(
    elementId TEXT,
    elementPolygon geometry,
    location TEXT,
    PRIMARY KEY (elementId)
);
"""

CREATE_SEGMENT_SQL = """
CREATE TABLE IF NOT EXISTS Segment(
    segmentId SERIAL,
    elementId TEXT,
    startPoint geometry,
    endPoint geometry,
    segmentLine geometry,
    heading real,
    FOREIGN KEY(elementId)
        REFERENCES SegmentPolygon(elementId)
);
"""

CREATE_SEGMENT_INDEX = """
    CREATE INDEX IF NOT EXISTS segPoint_idx
    ON Segment
    USING GiST(segmentLine);
"""

CREATE_LANESECTION_SQL = """
CREATE TABLE IF NOT EXISTS LaneSection(
    id Text,
    laneToLeft Text,
    laneToRight Text,
    fasterLane Text,
    slowerLane Text,
    isForward boolean,
    PRIMARY KEY (id),
    FOREIGN KEY(id)
        REFERENCES SegmentPolygon(elementId)
);
"""

CREATE_LANE_SQL = """
CREATE TABLE IF NOT EXISTS Lane(
    id Text,
    PRIMARY KEY (id),
    FOREIGN KEY(id)
        REFERENCES SegmentPolygon(elementId)
);
"""

CREATE_LANE_LANESEC_SQL = """
CREATE TABLE IF NOT EXISTS Lane_LaneSection(
    laneId TEXT,
    laneSectionId TEXT
);
"""

CREATE_LANEGROUP_SQL = """
CREATE TABLE IF NOT EXISTS LaneGroup(
    id Text,
    PRIMARY KEY (id),
    FOREIGN KEY(id)
        REFERENCES SegmentPolygon(elementId)
);
"""

CREATE_LANEGROUP_LANE_SQL = """
CREATE TABLE IF NOT EXISTS LaneGroup_Lane(
    laneGroupId TEXT,
    laneId TEXT
);
"""

CREATE_OPPOSITE_LANEGROUP_SQL = """
CREATE TABLE IF NOT EXISTS Opposite_LaneGroup(
    laneGroupId TEXT,
    oppositeId TEXT
);
"""

CREATE_ROAD_SQL = """
CREATE TABLE IF NOT EXISTS Road(
    id Text,
    forwardLane Text,
    backwardLane Text,
    PRIMARY KEY (id),
    FOREIGN KEY(id)
        REFERENCES SegmentPolygon(elementId)
);
"""

CREATE_ROAD_LANEGROUP_SQL = """
CREATE TABLE IF NOT EXISTS Road_LaneGroup(
    roadId TEXT,
    laneGroupId TEXT
);
"""

CREATE_ROAD_ROADSECTION_SQL = """
CREATE TABLE IF NOT EXISTS Road_RoadSection(
    roadId TEXT,
    roadSectionId TEXT
);
"""

CREATE_ROADSECTION_SQL = """
CREATE TABLE IF NOT EXISTS RoadSection(
    id TEXT,
    forwardLanes text[],
    backwardLanes text[],
    FOREIGN KEY(id)
        REFERENCES SegmentPolygon(elementId)
);
"""

CREATE_ROADSEC_LANESEC_SQL = """
CREATE TABLE IF NOT EXISTS RoadSection_LaneSection(
    roadSectionId TEXT,
    laneSectionId TEXT
);
"""

CREATE_INTERSECTION_SQL = """
CREATE TABLE IF NOT EXISTS Intersection(
    id TEXT,
    road TEXT,
    PRIMARY KEY (id),
    FOREIGN KEY(id)
        REFERENCES SegmentPolygon(elementId)
);
"""


def _remove_suffix(uid: str) -> "str | None":
    if uid is None:
        return None

    split = uid.split("_")
    assert len(split) == 2, f"cannot remove suffix: {uid}"
    return split[0]


def create_polygon_table(database: "Database", polygons, drop=True):
    if drop:
        database.update("DROP TABLE IF EXISTS SegmentPolygon CASCADE")
    database.update(CREATE_POLYGON_SQL)
    database.update("CREATE INDEX IF NOT EXISTS element_idx ON SegmentPolygon(elementId);")
    database.update("CREATE INDEX IF NOT EXISTS location_idx ON SegmentPolygon(location);")
    database.update(
        "CREATE INDEX IF NOT EXISTS segPoly_idx ON SegmentPolygon USING GiST(elementPolygon);"
    )

    ids = set([p["id"].split("_")[0] for p in polygons if len(p["id"].split("_")) == 1])

    values = []
    for poly in polygons:
        i = poly["id"]
        if len(i.split("_")) != 1:
            assert i.split("_")[0] in ids
            continue
        values.append(
            f"""(
                '{poly['id']}',
                '{poly['polygon']}',
                '{poly['location']}'
            )"""
        )

    if len(values):
        database.update(
            f"""
            INSERT INTO SegmentPolygon (
                elementId,
                elementPolygon,
                location
            )
            VALUES {','.join(values)};
            """
        )


def create_segment_table(database: "Database", segments, drop=True):
    if drop:
        database.update("DROP TABLE IF EXISTS Segment")
    database.update(CREATE_SEGMENT_SQL)
    database.update("CREATE INDEX IF NOT EXISTS element_idx ON Segment(elementId);")

    ids = set(
        [s["polygonId"].split("_")[0] for s in segments if len(s["polygonId"].split("_")) == 1]
    )

    values = []
    for seg in segments:
        i = seg["polygonId"]
        if len(i.split("_")) != 1:
            assert i.split("_")[0] in ids
            continue
        values.append(
            f"""(
                '{seg['polygonId']}',
                '{seg['start']}',
                '{seg['end']}',
                '{seg['heading']}'
            )"""
        )

    if len(values):
        database.update(
            f"""
            INSERT INTO Segment (
                elementId,
                startPoint,
                endPoint,
                heading
            )
            VALUES {','.join(values)};
            """
        )
    database.update(
        """
        UPDATE Segment
        SET segmentLine = ST_MakeLine(startPoint, endPoint)
        WHERE startPoint IS NOT NULL and endPoint IS NOT NULL;
        """
    )
    database.update(CREATE_SEGMENT_INDEX)


def create_lanesection_table(database: "Database", laneSections, drop=True):
    if drop:
        database.update("DROP TABLE IF EXISTS LaneSection")
    database.update(CREATE_LANESECTION_SQL)
    database.update("CREATE INDEX IF NOT EXISTS lanesec_idx ON LaneSection(id);")

    values = []
    for lanesec in laneSections:
        values.append(
            f"""(
                '{_remove_suffix(lanesec['id'])}',
                '{_remove_suffix(lanesec['laneToLeft'])}',
                '{_remove_suffix(lanesec['laneToRight'])}',
                '{lanesec['fasterLane']}',
                '{lanesec['slowerLane']}',
                '{lanesec['isForward']}'
            )"""
        )

    if len(values):
        database.update(
            f"""
            INSERT INTO LaneSection (
                id,
                laneToLeft,
                laneToRight,
                fasterLane,
                slowerLane,
                isForward
            )
            VALUES {','.join(values)};
            """
        )


def create_lane_table(database: "Database", lanes, drop=True):
    if drop:
        database.update("DROP TABLE IF EXISTS Lane")
    database.update(CREATE_LANE_SQL)
    database.update("CREATE INDEX IF NOT EXISTS lane_idx ON Lane(id);")

    values = []
    for lane in lanes:
        values.append(
            f"""(
                '{lane['id']}'
            )"""
        )

    if len(values):
        database.update(
            f"""
            INSERT INTO Lane (
                id
            )
            VALUES {','.join(values)};
            """
        )


def create_lane_lanesec_table(database: "Database", lane_lanesec, drop=True):
    if drop:
        database.update("DROP TABLE IF EXISTS Lane_LaneSection")
    database.update(CREATE_LANE_LANESEC_SQL)
    database.update("CREATE INDEX IF NOT EXISTS lane_idx ON Lane_LaneSection(laneId);")
    database.update(
        "CREATE INDEX IF NOT EXISTS laneSection_idx ON Lane_LaneSection(laneSectionId);"
    )

    values = []
    for ll in lane_lanesec:
        values.append(
            f"""(
                '{ll['lane']}',
                '{_remove_suffix(ll['laneSec'])}'
            )"""
        )

    if len(values):
        database.update(
            f"""
            INSERT INTO Lane_LaneSection (
                laneId,
                laneSectionId
            )
            VALUES {','.join(values)};
            """
        )


def create_lanegroup_table(database: "Database", laneGroups, drop=True):
    if drop:
        database.update("DROP TABLE IF EXISTS LaneGroup")
    database.update(CREATE_LANEGROUP_SQL)
    database.update("CREATE INDEX IF NOT EXISTS lanegroup_idx ON LaneGroup(id);")

    values = []
    for lg in laneGroups:
        values.append(f"('{lg['id']}')")

    if len(values):
        database.update(
            f"""
            INSERT INTO LaneGroup (id)
            VALUES {','.join(values)};
            """
        )


def create_lanegroup_lane_table(database: "Database", lanegroup_lane, drop=True):
    if drop:
        database.update("DROP TABLE IF EXISTS LaneGroup_Lane")
    database.update(CREATE_LANEGROUP_LANE_SQL)
    database.update("CREATE INDEX IF NOT EXISTS lane_idx ON LaneGroup_Lane(laneId);")
    database.update("CREATE INDEX IF NOT EXISTS laneGroup_idx ON LaneGroup_Lane(laneGroupId);")

    values = []
    for ll in lanegroup_lane:
        values.append(
            f"""(
                '{ll['laneGroup']}',
                '{ll['lane']}'
            )"""
        )

    if len(values):
        database.update(
            f"""
            INSERT INTO LaneGroup_Lane (
                laneGroupId,
                laneId
            )
            VALUES {','.join(values)};
            """
        )


def create_opposite_lanegroup_table(database: "Database", opposite_lanegroup, drop=True):
    if drop:
        database.update("DROP TABLE IF EXISTS Opposite_LaneGroup")
    database.update(CREATE_OPPOSITE_LANEGROUP_SQL)
    database.update("CREATE INDEX IF NOT EXISTS opposite_idx ON Opposite_LaneGroup(oppositeId);")
    database.update("CREATE INDEX IF NOT EXISTS laneGroup_idx ON Opposite_LaneGroup(laneGroupId);")

    values = []
    for oppo in opposite_lanegroup:
        values.append(
            f"""(
                '{oppo['lane']}',
                '{oppo['opposite']}'
            )"""
        )

    if len(values):
        database.update(
            f"""
            INSERT INTO Opposite_LaneGroup (
                laneGroupId,
                oppositeId
            )
            VALUES {','.join(values)};
            """
        )


def create_road_table(database: "Database", roads, drop=True):
    if drop:
        database.update("DROP TABLE IF EXISTS Road")
    database.update(CREATE_ROAD_SQL)
    database.update("CREATE INDEX IF NOT EXISTS road_idx ON Road(id);")

    values = []
    for road in roads:
        values.append(
            f"""(
                '{road['id']}',
                '{road['forwardLanes']}',
                '{road['backwardLanes']}'
            )"""
        )

    if len(values):
        database.update(
            f"""
            INSERT INTO Road (
                id,
                forwardLane,
                backwardLane
            )
            VALUES {','.join(values)};
            """
        )


def create_road_lanegroup_table(database: "Database", road_lanegroup, drop=True):
    if drop:
        database.update("DROP TABLE IF EXISTS Road_LaneGroup")
    database.update(CREATE_ROAD_LANEGROUP_SQL)
    database.update("CREATE INDEX IF NOT EXISTS road_idx ON Road_LaneGroup(roadId);")
    database.update("CREATE INDEX IF NOT EXISTS laneGroup_idx ON Road_LaneGroup(laneGroupId);")

    values = []
    for rl in road_lanegroup:
        values.append(
            f"""(
                '{rl['road']}',
                '{rl['laneGroup']}'
            )"""
        )

    if len(values):
        database.update(
            f"""
            INSERT INTO Road_LaneGroup (
                roadId,
                laneGroupId
            )
            VALUES {','.join(values)};
            """
        )


def create_road_roadsec_table(database: "Database", road_roadsec, drop=True):
    if drop:
        database.update("DROP TABLE IF EXISTS Road_RoadSection")
    database.update(CREATE_ROAD_ROADSECTION_SQL)
    database.update("CREATE INDEX IF NOT EXISTS road_idx ON Road_RoadSection(roadId);")
    database.update("CREATE INDEX IF NOT EXISTS roadsec_idx ON Road_RoadSection(roadSectionId);")

    values = []
    for rr in road_roadsec:
        values.append(
            f"""(
                '{rr['road']}',
                '{_remove_suffix(rr['roadSec'])}'
            )"""
        )

    if len(values):
        database.update(
            f"""
            INSERT INTO Road_RoadSection (
                roadId,
                roadSectionId
            )
            VALUES {','.join(values)};
            """
        )


def create_roadsection_table(database: "Database", roadSections, drop=True):
    if drop:
        database.update("DROP TABLE IF EXISTS RoadSection")
    database.update(CREATE_ROADSECTION_SQL)
    database.update("CREATE INDEX IF NOT EXISTS roadsec_idx ON RoadSection(id);")

    values = []
    for roadsec in roadSections:
        if len(roadsec["forwardLanes"]) == 0:
            fl = "[]::text[]"
        else:
            fl = str([*map(_remove_suffix, roadsec["forwardLanes"])])

        if len(roadsec["backwardLanes"]) == 0:
            bl = "[]::text[]"
        else:
            bl = str([*map(_remove_suffix, roadsec["backwardLanes"])])

        values.append(
            f"""(
                '{_remove_suffix(roadsec['id'])}',
                ARRAY{fl},
                ARRAY{bl}
            )"""
        )

    if len(values):
        database.update(
            f"""
            INSERT INTO RoadSection (
                id,
                forwardLanes,
                backwardLanes
            )
            VALUES {','.join(values)};
            """
        )


def create_roadsec_lanesec_table(database: "Database", roadsec_lanesec, drop=True):
    if drop:
        database.update("DROP TABLE IF EXISTS RoadSection_LaneSection")
    database.update(CREATE_ROADSEC_LANESEC_SQL)
    database.update(
        "CREATE INDEX IF NOT EXISTS lanesec_idx ON RoadSection_LaneSection(laneSectionId);"
    )
    database.update(
        "CREATE INDEX IF NOT EXISTS roadsec_idx ON RoadSection_LaneSection(roadSectionId);"
    )

    values = []
    for rl in roadsec_lanesec:
        values.append(
            f"""(
                '{_remove_suffix(rl['roadSec'])}',
                '{_remove_suffix(rl['laneSec'])}'
            )"""
        )

    if len(values):
        database.update(
            f"""
            INSERT INTO RoadSection_LaneSection (
                roadSectionId,
                laneSectionId
            )
            VALUES {','.join(values)};
            """
        )


def create_intersection_table(database: "Database", intersections, drop=True):
    if drop:
        database.update("DROP TABLE IF EXISTS Intersection")
    database.update(CREATE_INTERSECTION_SQL)
    database.update("CREATE INDEX IF NOT EXISTS intersec_idx ON Intersection(id);")

    values = []
    for intersec in intersections:
        values.append(
            f"""(
                '{_remove_suffix(intersec['id'])}',
                '{intersec['road']}'
            )"""
        )

    if len(values):
        database.update(
            f"""
            INSERT INTO Intersection (
                id,
                road
            )
            VALUES {','.join(values)};
            """
        )


ROAD_TYPES = {"road", "lane", "lanesection", "roadsection", "intersection", "lanegroup"}


def add_segment_type(database: "Database"):
    database.update("ALTER TABLE SegmentPolygon ADD segmentTypes text[];")
    print("altered table")
    for road_type in ROAD_TYPES:
        # Add road type to all polygons of that type
        query = f"""UPDATE SegmentPolygon
                SET segmentTypes = ARRAY_APPEND(segmentTypes, '{road_type}')
                WHERE elementId IN (SELECT id FROM {road_type});"""
        database.update(query)
        print("added type:", road_type)


CREATE_TABLES = {
    "polygon": create_polygon_table,
    "segment": create_segment_table,
    "laneSection": create_lanesection_table,
    "lane": create_lane_table,
    "lane_LaneSec": create_lane_lanesec_table,
    "laneGroup": create_lanegroup_table,
    "laneGroup_Lane": create_lanegroup_lane_table,
    "laneGroup_opposite": create_opposite_lanegroup_table,
    "road": create_road_table,
    "road_laneGroup": create_road_lanegroup_table,
    "road_roadSec": create_road_roadsec_table,
    "roadSection": create_roadsection_table,
    "roadSec_laneSec": create_roadsec_lanesec_table,
    "intersection": create_intersection_table,
}


def ingest_location(database: "Database", directory: "str", location: "str"):
    print("Location:", location)
    filenames = os.listdir(directory)

    assert set(filenames) == set([k + ".json" for k in CREATE_TABLES.keys()]), (
        sorted(filenames),
        sorted([k + ".json" for k in CREATE_TABLES.keys()]),
    )

    for d, fn in CREATE_TABLES.items():
        with open(os.path.join(directory, d + ".json"), "r") as f:
            data = json.load(f)

        print("Ingesting", d)
        fn(database, [{"location": location, **d} for d in data], drop=False)


def ingest_road(database: "Database", directory: str):
    # Create tables
    for d, fn in CREATE_TABLES.items():
        print("Create Table", d)
        fn(database, [])

    filenames = os.listdir(directory)

    if all(os.path.isdir(os.path.join(directory, f)) for f in filenames):
        for d in filenames:
            if d == "boston-old":
                continue

            print(d)
            ingest_location(database, os.path.join(directory, d), d)
    else:
        if any(os.path.isdir(os.path.join(directory, f)) for f in filenames):
            raise Exception()

        ingest_location(database, directory, "boston-seaport")

    print("adding segment types")
    add_segment_type(database)
