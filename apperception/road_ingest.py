import psycopg2

conn = psycopg2.connect(
    database="mobilitydb", user="docker", password="docker", host="localhost", port=25432
)

CREATE_POLYGON_SQL = """
CREATE TABLE IF NOT EXISTS Polygon(
    elementId TEXT,
    elementPolygon geometry,
    PRIMARY KEY (elementId)
);
"""

CREATE_SEGMENT_SQL = """
CREATE TABLE IF NOT EXISTS Segment(
    segmentId SERIAL,
    elementId TEXT,
    startPoint geometry,
    endPoint geometry,
    heading real,
    FOREIGN KEY(elementId)
        REFERENCES Polygon(elementId)
);
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
        REFERENCES Polygon(elementId)
);
"""

CREATE_LANE_SQL = """
CREATE TABLE IF NOT EXISTS Lane(
    id Text,
    PRIMARY KEY (id),
    FOREIGN KEY(id)
        REFERENCES Polygon(elementId)
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
        REFERENCES Polygon(elementId)
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
        REFERENCES Polygon(elementId)
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
        REFERENCES Polygon(elementId)
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
        REFERENCES Polygon(elementId)
);
"""


def create_polygon_table(polygons, drop=True):
    cursor = conn.cursor()
    if drop:
        cursor.execute("DROP TABLE IF EXISTS Polygon CASCADE")
    cursor.execute(CREATE_POLYGON_SQL)
    cursor.execute("CREATE INDEX IF NOT EXISTS element_idx ON Polygon(elementId);")

    values = []
    for poly in polygons:
        values.append(
            f"""(
                '{poly['id']}',
                '{poly['polygon']}'
            )"""
        )

    cursor.execute(
        f"""
        INSERT INTO Polygon (
            elementId,
            elementPolygon
        )
        VALUES {','.join(values)};
        """
    )

    conn.commit()


def create_segment_table(segments, drop=True):
    cursor = conn.cursor()
    if drop:
        cursor.execute("DROP TABLE IF EXISTS Segment")
    cursor.execute(CREATE_SEGMENT_SQL)
    cursor.execute("CREATE INDEX IF NOT EXISTS element_idx ON Segment(elementId);")

    values = []
    for seg in segments:
        values.append(
            f"""(
                '{seg['polygonId']}',
                '{seg['start']}',
                '{seg['end']}',
                '{seg['heading']}'
            )"""
        )

    cursor.execute(
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

    conn.commit()


def create_lanesection_table(laneSections, drop=True):
    cursor = conn.cursor()
    if drop:
        cursor.execute("DROP TABLE IF EXISTS LaneSection")
    cursor.execute(CREATE_LANESECTION_SQL)
    cursor.execute("CREATE INDEX IF NOT EXISTS lanesec_idx ON LaneSection(id);")

    values = []
    for lanesec in laneSections:
        values.append(
            f"""(
                '{lanesec['id']}',
                '{lanesec['laneToLeft']}',
                '{lanesec['laneToRight']}',
                '{lanesec['fasterLane']}',
                '{lanesec['slowerLane']}',
                '{lanesec['isForward']}'
            )"""
        )

    cursor.execute(
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

    conn.commit()


def create_lane_table(lanes, drop=True):
    cursor = conn.cursor()
    if drop:
        cursor.execute("DROP TABLE IF EXISTS Lane")
    cursor.execute(CREATE_LANE_SQL)
    cursor.execute("CREATE INDEX IF NOT EXISTS lane_idx ON Lane(id);")

    values = []
    for lane in lanes:
        values.append(
            f"""(
                '{lane['id']}'
            )"""
        )

    cursor.execute(
        f"""
        INSERT INTO Lane (
            id
        )
        VALUES {','.join(values)};
        """
    )

    conn.commit()


def create_lane_lanesec_table(lane_lanesec, drop=True):
    cursor = conn.cursor()
    if drop:
        cursor.execute("DROP TABLE IF EXISTS Lane_LaneSection")
    cursor.execute(CREATE_LANE_LANESEC_SQL)
    cursor.execute("CREATE INDEX IF NOT EXISTS lane_idx ON Lane_LaneSection(laneId);")
    cursor.execute("CREATE INDEX IF NOT EXISTS laneSection_idx ON Lane_LaneSection(laneSectionId);")

    values = []
    for ll in lane_lanesec:
        values.append(
            f"""(
                '{ll['lane']}',
                '{ll['laneSec']}'
            )"""
        )

    cursor.execute(
        f"""
        INSERT INTO Lane_LaneSection (
            laneId,
            laneSectionId
        )
        VALUES {','.join(values)};
        """
    )

    conn.commit()


def create_lanegroup_table(laneGroups, drop=True):
    cursor = conn.cursor()
    if drop:
        cursor.execute("DROP TABLE IF EXISTS LaneGroup")
    cursor.execute(CREATE_LANEGROUP_SQL)
    cursor.execute("CREATE INDEX IF NOT EXISTS lanegroup_idx ON LaneGroup(id);")

    values = []
    for lg in laneGroups:
        values.append(f"('{lg['id']}')")

    cursor.execute(
        f"""
        INSERT INTO LaneGroup (id)
        VALUES {','.join(values)};
        """
    )

    conn.commit()


def create_lanegroup_lane_table(lanegroup_lane, drop=True):
    cursor = conn.cursor()
    if drop:
        cursor.execute("DROP TABLE IF EXISTS LaneGroup_Lane")
    cursor.execute(CREATE_LANEGROUP_LANE_SQL)
    cursor.execute("CREATE INDEX IF NOT EXISTS lane_idx ON LaneGroup_Lane(laneId);")
    cursor.execute("CREATE INDEX IF NOT EXISTS laneGroup_idx ON LaneGroup_Lane(laneGroupId);")

    values = []
    for ll in lanegroup_lane:
        values.append(
            f"""(
                '{ll['laneGroup']}',
                '{ll['lane']}'
            )"""
        )

    cursor.execute(
        f"""
        INSERT INTO LaneGroup_Lane (
            laneGroupId,
            laneId
        )
        VALUES {','.join(values)};
        """
    )

    conn.commit()


def create_opposite_lanegroup_table(opposite_lanegroup, drop=True):
    cursor = conn.cursor()
    if drop:
        cursor.execute("DROP TABLE IF EXISTS Opposite_LaneGroup")
    cursor.execute(CREATE_OPPOSITE_LANEGROUP_SQL)
    cursor.execute("CREATE INDEX IF NOT EXISTS opposite_idx ON Opposite_LaneGroup(oppositeId);")
    cursor.execute("CREATE INDEX IF NOT EXISTS laneGroup_idx ON Opposite_LaneGroup(laneGroupId);")

    values = []
    for oppo in opposite_lanegroup:
        values.append(
            f"""(
                '{oppo['lane']}',
                '{oppo['opposite']}'
            )"""
        )

    cursor.execute(
        f"""
        INSERT INTO Opposite_LaneGroup (
            laneGroupId,
            oppositeId
        )
        VALUES {','.join(values)};
        """
    )

    conn.commit()


def create_road_table(roads, drop=True):
    cursor = conn.cursor()
    if drop:
        cursor.execute("DROP TABLE IF EXISTS Road")
    cursor.execute(CREATE_ROAD_SQL)
    cursor.execute("CREATE INDEX IF NOT EXISTS road_idx ON Road(id);")

    values = []
    for road in roads:
        values.append(
            f"""(
                '{road['id']}',
                '{road['forwardLanes']}',
                '{road['backwardLanes']}'
            )"""
        )

    cursor.execute(
        f"""
        INSERT INTO Road (
            id,
            forwardLane,
            backwardLane
        )
        VALUES {','.join(values)};
        """
    )

    conn.commit()


def create_road_lanegroup_table(road_lanegroup, drop=True):
    cursor = conn.cursor()
    if drop:
        cursor.execute("DROP TABLE IF EXISTS Road_LaneGroup")
    cursor.execute(CREATE_ROAD_LANEGROUP_SQL)
    cursor.execute("CREATE INDEX IF NOT EXISTS road_idx ON Road_LaneGroup(roadId);")
    cursor.execute("CREATE INDEX IF NOT EXISTS laneGroup_idx ON Road_LaneGroup(laneGroupId);")

    values = []
    for rl in road_lanegroup:
        values.append(
            f"""(
                '{rl['road']}',
                '{rl['laneGroup']}'
            )"""
        )

    cursor.execute(
        f"""
        INSERT INTO Road_LaneGroup (
            roadId,
            laneGroupId
        )
        VALUES {','.join(values)};
        """
    )

    conn.commit()


def create_road_roadsec_table(road_roadsec, drop=True):
    cursor = conn.cursor()
    if drop:
        cursor.execute("DROP TABLE IF EXISTS Road_RoadSection")
    cursor.execute(CREATE_ROAD_ROADSECTION_SQL)
    cursor.execute("CREATE INDEX IF NOT EXISTS road_idx ON Road_RoadSection(roadId);")
    cursor.execute("CREATE INDEX IF NOT EXISTS roadsec_idx ON Road_RoadSection(roadSectionId);")

    values = []
    for rr in road_roadsec:
        values.append(
            f"""(
                '{rr['road']}',
                '{rr['roadSec']}'
            )"""
        )

    cursor.execute(
        f"""
        INSERT INTO Road_RoadSection (
            roadId,
            roadSectionId
        )
        VALUES {','.join(values)};
        """
    )

    conn.commit()


def create_roadsection_table(roadSections, drop=True):
    cursor = conn.cursor()
    if drop:
        cursor.execute("DROP TABLE IF EXISTS RoadSection")
    cursor.execute(CREATE_ROADSECTION_SQL)
    cursor.execute("CREATE INDEX IF NOT EXISTS roadsec_idx ON RoadSection(id);")

    values = []
    for roadsec in roadSections:
        if len(roadsec["forwardLanes"]) == 0:
            roadsec["forwardLanes"] = "[]::text[]"
        if len(roadsec["backwardLanes"]) == 0:
            roadsec["backwardLanes"] = "[]::text[]"

        values.append(
            f"""(
                '{roadsec['id']}',
                ARRAY{roadsec['forwardLanes']},
                ARRAY{roadsec['backwardLanes']}
            )"""
        )

    cursor.execute(
        f"""
        INSERT INTO RoadSection (
            id,
            forwardLanes,
            backwardLanes
        )
        VALUES {','.join(values)};
        """
    )

    conn.commit()


def create_roadsec_lanesec_table(roadsec_lanesec, drop=True):
    cursor = conn.cursor()
    if drop:
        cursor.execute("DROP TABLE IF EXISTS RoadSection_LaneSection")
    cursor.execute(CREATE_ROADSEC_LANESEC_SQL)
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS lanesec_idx ON RoadSection_LaneSection(laneSectionId);"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS roadsec_idx ON RoadSection_LaneSection(roadSectionId);"
    )

    values = []
    for rl in roadsec_lanesec:
        values.append(
            f"""(
                '{rl['roadSec']}',
                '{rl['laneSec']}'
            )"""
        )

    cursor.execute(
        f"""
        INSERT INTO RoadSection_LaneSection (
            roadSectionId,
            laneSectionId
        )
        VALUES {','.join(values)};
        """
    )

    conn.commit()


def create_intersection_table(intersections, drop=True):
    cursor = conn.cursor()
    if drop:
        cursor.execute("DROP TABLE IF EXISTS Intersection")
    cursor.execute(CREATE_INTERSECTION_SQL)
    cursor.execute("CREATE INDEX IF NOT EXISTS intersec_idx ON Intersection(id);")

    values = []
    for intersec in intersections:
        values.append(
            f"""(
                '{intersec['id']}',
                '{intersec['road']}'
            )"""
        )

    cursor.execute(
        f"""
        INSERT INTO Intersection (
            id,
            road
        )
        VALUES {','.join(values)};
        """
    )

    conn.commit()
