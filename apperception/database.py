from os import environ
from typing import TYPE_CHECKING, Callable, List, Tuple

import pandas as pd
import psycopg2
import psycopg2.errors
import psycopg2.sql as psql
from mobilitydb.psycopg import register as mobilitydb_register
from postgis.psycopg import register as postgis_register
from datetime import datetime
from apperception.data_types import Trajectory
from apperception.predicate import (
    FindAllTablesVisitor,
    GenSqlVisitor,
    MapTablesTransformer,
    normalize,
)
from apperception.utils.add_recognized_objects import add_recognized_objects
from apperception.utils.create_sql import create_sql
from apperception.utils.fetch_camera import fetch_camera
from apperception.utils.fetch_camera_framenum import fetch_camera_framenum
from apperception.utils.overlay_bboxes import overlay_bboxes
from apperception.utils.recognize import recognize
from apperception.utils.reformat_bbox_trajectories import reformat_bbox_trajectories
from apperception.utils.timestamp_to_framenum import timestamp_to_framenum

if TYPE_CHECKING:
    from psycopg2 import connection as Connection
    from psycopg2 import cursor as Cursor

    from .data_types import Camera
    from .predicate import PredicateNode
    from .world import World

CAMERA_TABLE = "Cameras"
TRAJ_TABLE = "Item_General_Trajectory"
BBOX_TABLE = "General_Bbox"

CAMERA_COLUMNS: "list[tuple[str, str]]" = [
    ("cameraId", "TEXT"),
    ("frameId", "TEXT"),
    ("frameNum", "Int"),
    ("fileName", "TEXT"),
    ("cameraTranslation", "geometry"),
    ("cameraRotation", "real[4]"),
    ("cameraIntrinsic", "real[3][3]"),
    ("egoTranslation", "geometry"),
    ("egoRotation", "real[4]"),
    ("timestamp", "timestamptz"),
    ("cameraHeading", "real"),
    ("egoHeading", "real"),
]

TRAJECTORY_COLUMNS: "list[tuple[str, str]]" = [
    ("itemId", "TEXT"),
    ("cameraId", "TEXT"),
    ("objectType", "TEXT"),
    # ("roadTypes", "ttext"),
    ("trajCentroids", "tgeompoint"),
    ("translations", "tgeompoint"),  # [(x,y,z)@today, (x2, y2,z2)@tomorrow, (x2, y2,z2)@nextweek]
    ("itemHeadings", "tfloat"),
    # ("roadPolygons", "tgeompoint"),
    # ("period", "period") [today, nextweek]
]

BBOX_COLUMNS: "list[tuple[str, str]]" = [
    ("itemId", "TEXT"),
    ("cameraId", "TEXT"),
    ("trajBbox", "stbox"),
    ("timestamp", "timestamptz"),
]


def columns(fn: Callable[[Tuple[str, str]], str], columns: List[Tuple[str, str]]) -> str:
    return ",".join(map(fn, columns))


def _schema(column: Tuple[str, str]) -> str:
    return " ".join(column)


def _name(column: Tuple[str, str]) -> str:
    return column[0]


def place_holder(num: int):
    return ",".join(["%s"] * num)


class Database:
    connection: "Connection"
    cursor: "Cursor"

    def __init__(self, connection: "Connection"):
        self.connection = connection
        postgis_register(self.connection)
        mobilitydb_register(self.connection)
        self.cursor = self.connection.cursor()

    def reset(self, commit=False):
        self._create_camera_table(commit)
        self._create_item_general_trajectory_table(commit)
        # self._create_general_bbox_table(False)
        self._create_index(commit)

    def _create_camera_table(self, commit=True):
        self.cursor.execute("DROP TABLE IF EXISTS Cameras CASCADE;")
        self.cursor.execute(f"CREATE TABLE Cameras ({columns(_schema, CAMERA_COLUMNS)})")
        self._commit(commit)

    # def _create_general_bbox_table(self, commit=True):
    #     self.cursor.execute("DROP TABLE IF EXISTS General_Bbox CASCADE;")
    #     self.cursor.execute(
    #         f"""
    #         CREATE TABLE General_Bbox (
    #             {columns(_schema, BBOX_COLUMNS)},
    #             FOREIGN KEY(itemId) REFERENCES Item_General_Trajectory(itemId),
    #             PRIMARY KEY (itemId, timestamp)
    #         )
    #         """
    #     )
    #     self._commit(commit)

    def _create_item_general_trajectory_table(self, commit=True):
        self.cursor.execute("DROP TABLE IF EXISTS Item_General_Trajectory CASCADE;")
        self.cursor.execute(
            f"""
            CREATE TABLE Item_General_Trajectory (
                {columns(_schema, TRAJECTORY_COLUMNS)},
                PRIMARY KEY (itemId)
            )
            """
        )
        self._commit(commit)

    def _create_index(self, commit=True):
        self.cursor.execute("CREATE INDEX ON Cameras (cameraId);")
        self.cursor.execute("CREATE INDEX ON Cameras (timestamp);")
        self.cursor.execute("CREATE INDEX ON Item_General_Trajectory (itemId);")
        self.cursor.execute("CREATE INDEX ON Item_General_Trajectory (cameraId);")
        self.cursor.execute(
            "CREATE INDEX IF NOT EXISTS traj_idx ON Item_General_Trajectory USING GiST(trajCentroids);"
        )
        self.cursor.execute(
            "CREATE INDEX IF NOT EXISTS trans_idx ON Item_General_Trajectory USING GiST(translations);"
        )
        # self.cursor.execute("CREATE INDEX IF NOT EXISTS item_idx ON General_Bbox(itemId);")
        # self.cursor.execute(
        #     "CREATE INDEX IF NOT EXISTS traj_bbox_idx ON General_Bbox USING GiST(trajBbox);"
        # )
        # self.cursor.execute(
        #     "CREATE INDEX IF NOT EXISTS item_id_timestampx ON General_Bbox(itemId, timestamp);"
        # )
        self._commit(commit)

    def _insert_into_camera(self, value: tuple, commit=True):
        self.cursor.execute(
            f"INSERT INTO Cameras ({columns(_name, CAMERA_COLUMNS)}) VALUES ({place_holder(len(CAMERA_COLUMNS))})",
            tuple(value),
        )
        self._commit(commit)

    def _insert_into_item_general_trajectory(self, value: tuple, commit=True):
        self.cursor.execute(
            f"INSERT INTO Item_General_Trajectory ({columns(_name, TRAJECTORY_COLUMNS)}) VALUES ({place_holder(len(TRAJECTORY_COLUMNS))})",
            tuple(value),
        )
        self._commit(commit)

    # def _insert_into_general_bbox(self, value: tuple, commit=True):
    #     self.cursor.execute(
    #         f"INSERT INTO General_Bbox ({columns(_name, BBOX_COLUMNS)}) VALUES ({place_holder(len(BBOX_COLUMNS))})",
    #         tuple(value),
    #     )
    #     self._commit(commit)

    def _commit(self, commit=True):
        if commit:
            self.connection.commit()

    def execute(
        self, query: "str | psql.Composable", vars: "tuple | list | None" = None
    ) -> "list[tuple]":
        try:
            self.cursor.execute(query, vars)
            for notice in self.cursor.connection.notices:
                print(notice)
            if self.cursor.pgresult_ptr is not None:
                return self.cursor.fetchall()
            else:
                return []
        except psycopg2.errors.DatabaseError as error:
            self.connection.rollback()
            raise error

    def update(self, query: "str | psql.Composable", commit: bool = True) -> None:
        try:
            self.cursor.execute(query)
            self._commit(commit)
        except psycopg2.errors.DatabaseError as error:
            self.connection.rollback()
            raise error
    
    def insert_cam(self, camera: "Camera"):
        values = [
            f"""(
                '{camera.id}',
                '{config.frame_id}',
                {config.frame_num},
                '{config.filename}',
                'POINT Z ({' '.join(map(str, config.camera_translation))})',
                ARRAY[{','.join(map(str, config.camera_rotation))}]::real[],
                ARRAY{config.camera_intrinsic}::real[][],
                'POINT Z ({' '.join(map(str, config.ego_translation))})',
                ARRAY[{','.join(map(str, config.ego_rotation))}]::real[],
                '{datetime.fromtimestamp(float(config.timestamp)/1000000.0)}',
                {config.cameraHeading},
                {config.egoHeading}
            )"""
            for config in camera.configs
        ]

        self.cursor.execute(
            f"""
            INSERT INTO Cameras ({",".join(col for col, _ in CAMERA_COLUMNS)})
            VALUES {','.join(values)};
            """
        )

        # print("New camera inserted successfully.........")
        self.connection.commit()

    def retrieve_cam(self, query: "psql.Composed | str | None" = None, camera_id: str = ""):
        """
        Called when executing update commands (add_camera, add_objs ...etc)
        """

        q = self._select_cam_with_camera_id(camera_id)
        return (psql.SQL("({}) UNION ({})").format(create_sql(query), q) if query else q).as_string(
            self.cursor
        )  # UNION

    def _select_cam_with_camera_id(self, camera_id: str):
        """
        Select cams with certain world id
        """
        return psql.SQL(f"SELECT * FROM Cameras WHERE cameraId = {camera_id}").format(
            camera_id=camera_id
        )

    def filter(self, query: "psql.Composable | str", predicate: "PredicateNode"):
        tables, camera = FindAllTablesVisitor()(predicate)
        tables = sorted(tables)
        mapping = {t: i for i, t in enumerate(tables)}
        predicate = normalize(predicate)
        predicate = MapTablesTransformer(mapping)(predicate)
        query_str = query if isinstance(query, str) else query.as_string(self.cursor)
        joins = [f"JOIN ({query_str}) as t{i} USING (cameraId)" for i in range(1, len(tables))]

        return f"""
        SELECT DISTINCT *
        FROM ({query_str}) as t0
        {" ".join(joins)}
        {f"JOIN Cameras USING (cameraId)" if camera else ""}
        WHERE {GenSqlVisitor()(predicate)}
        """

    def exclude(self, query: "psql.Composable | str", world: "World"):
        query_str = query if isinstance(query, str) else query.as_string(self.cursor)
        return f"""
        SELECT *
        FROM ({query_str}) as __query__
        EXCEPT
        SELECT *
        FROM ({world._execute_from_root()}) as __except__
        """

    def union(self, query: "psql.Composable | str", world: "World"):
        query_str = query if isinstance(query, str) else query.as_string(self.cursor)
        return f"""
        SELECT *
        FROM ({query_str}) as __query__
        UNION
        SELECT *
        FROM ({world._execute_from_root()}) as __union__
        """

    def intersect(self, query: "psql.Composable | str", world: "World"):
        query_str = query if isinstance(query, str) else query.as_string(self.cursor)
        return f"""
        SELECT *
        FROM ({query_str}) as __query__
        INTERSECT
        SELECT *
        FROM ({world._execute_from_root()}) as __intersect__
        """

    def get_cam(self, query: "psql.Composable | str"):
        """
        Execute sql command rapidly
        """

        # hack
        q = psql.SQL(
            "SELECT cameraID, frameId, frameNum, fileName, "
            "cameraTranslation, cameraRotation, cameraIntrinsic, "
            "egoTranslation, egoRotation, timestamp, cameraHeading, egoHeading "
            "FROM ({query}) AS final"
        ).format(query=create_sql(query))
        return self.execute(q)

    def fetch_camera(self, scene_name: str, frame_timestamp: List[str]):
        return fetch_camera(self.connection, scene_name, frame_timestamp)

    def fetch_camera_framenum(self, scene_name: str, frame_num: List[int]):
        return fetch_camera_framenum(self.connection, scene_name, frame_num)

    def timestamp_to_framenum(self, scene_name: str, timestamps: List[str]):
        return timestamp_to_framenum(self.connection, scene_name, timestamps)

    def insert_bbox_traj(self, camera: "Camera", annotation):
        tracking_results = recognize(camera.configs, annotation)
        add_recognized_objects(self.connection, tracking_results, camera.id)

    def retrieve_bbox(self, query: "psql.Composable | str | None" = None, camera_id: str = ""):
        q = psql.SQL("SELECT * FROM General_Bbox WHERE cameraId = {camera_id}").format(
            camera_id=camera_id
        )
        return (psql.SQL("({}) UNION ({})").format(create_sql(query), q) if query else q).as_string(
            self.cursor
        )

    def retrieve_traj(self, query: "psql.Composable | str | None" = None, camera_id: str = ""):
        q = psql.SQL("SELECT * FROM Item_General_Trajectory WHERE cameraId = {camera_id}").format(
            camera_id=camera_id
        )
        return (psql.SQL("({}) UNION ({})").format(create_sql(query), q) if query else q).as_string(
            self.cursor
        )

    def road_direction(self, x: float, y: float, default_dir: float):
        return self.execute(f"SELECT roadDirection({x}, {y}, {default_dir});")

    def road_coords(self, x: float, y: float):
        return self.execute(f"SELECT roadCoords({x}, {y});")

    def select_all(self, query: "psql.Composable | str") -> List[tuple]:
        print("select_all:", query if isinstance(query, str) else query.as_string(self.cursor))
        return self.execute(query)

    def get_traj(self, query: "psql.Composable | str") -> List[List[Trajectory]]:
        # hack
        _query = psql.SQL(
            "SELECT asMFJSON(trajCentroids)::json->'sequences'" "FROM ({query}) as final"
        ).format(query=query)

        print("get_traj", _query.as_string(self.cursor))
        trajectories = self.execute(_query)
        return [
            [
                Trajectory(
                    coordinates=t["coordinates"],
                    datetimes=t["datetimes"],
                    lower_inc=t["lower_inc"],
                    upper_inc=t["upper_inc"],
                )
                for t in trajectory
            ]
            for (trajectory,) in trajectories
        ]

    def get_traj_key(self, query: "psql.Composable | str"):
        _query = psql.SQL("SELECT itemId FROM ({query}) as final").format(query=create_sql(query))
        print("get_traj_key", _query.as_string(self.cursor))
        return self.execute(_query)

    def get_id_time_camId_filename(self, query: "psql.Composable | str", num_joined_tables: int):
        itemId = ",".join([f"t{i}.itemId" for i in range(num_joined_tables)])
        timestamp = "cameras.timestamp"
        camId = "cameras.cameraId"
        filename = "cameras.filename"
        _query = (
            create_sql(query)
            .as_string(self.cursor)
            .replace("SELECT DISTINCT *", f"SELECT {itemId}, {timestamp}, {camId}, {filename}", 1)
        )

        print("get_id_time_camId_filename", _query)
        return self.execute(_query)

    def get_video(self, query, cams, boxed):
        query = psql.SQL(
            "SELECT XMin(trajBbox), YMin(trajBbox), ZMin(trajBbox), "
            "XMax(trajBbox), YMax(trajBbox), ZMax(trajBbox), TMin(trajBbox) "
            "FROM ({query}) "
            "JOIN General_Bbox using (itemId)"
        )

        fetched_meta = self.execute(query)
        _fetched_meta = reformat_bbox_trajectories(fetched_meta)
        overlay_bboxes(_fetched_meta, cams, boxed)

    def sql(self, query: str) -> pd.DataFrame:
        return pd.DataFrame(self.execute(query), columns=[d.name for d in self.cursor.description])

    # def get_len(self, query: "psql.Composable | str"):
    #     """
    #     Execute sql command rapidly
    #     """

    #     # hack
    #     q = psql.SQL(
    #         "SELECT ratio, ST_X(origin), ST_Y(origin), "
    #         "ST_Z(origin), fov, skev_factor "
    #         "FROM ({query}) AS final"
    #     ).format(query=query)
    #     return self.execute(q)

    # def get_traj_attr(self, query: "psql.Composable | str", attr: str):
    #     _query = psql.SQL("SELECT {attr} FROM ({query}) as final").format(attr=attr, query=query)
    #     print("get_traj_attr:", attr, _query.as_string(self.cursor))
    #     return self.execute(_query)

    # def get_bbox_geo(self, query: "psql.Composable | str"):
    #     return self.execute(
    #         psql.SQL(
    #             "SELECT XMin(trajBbox), YMin(trajBbox), ZMin(trajBbox), "
    #             "XMax(trajBbox), YMax(trajBbox), ZMax(trajBbox) "
    #             "FROM ({query})"
    #         ).format(query=create_sql(query))
    #     )

    # def get_time(self, query: "psql.Composable | str"):
    #     return self.execute(
    #         psql.SQL("SELECT Tmin(trajBbox) FROM ({query})").format(query=create_sql(query))
    #     )

    # def get_distance(self, query: "psql.Composable | str", start: str, end: str):
    #     return self.execute(
    #         psql.SQL(
    #             "SELECT cumulativeLength(atPeriodSet(trajCentroids, {[{start}, {end})})) "
    #             "FROM ({query})"
    #         ).format(query=create_sql(query), start=psql.Literal(start), end=psql.Literal(end))
    #     )

    # def get_speed(self, query, start, end):
    #     return self.execute(
    #         psql.SQL(
    #             "SELECT speed(atPeriodSet(trajCentroids, {[{start}, {end})})) " "FROM ({query})"
    #         ).format(query=create_sql(query), start=psql.Literal(start), end=psql.Literal(end))
    #     )


database = Database(
    psycopg2.connect(
        dbname=environ.get("AP_DB", "mobilitydb"),
        user=environ.get("AP_USER", "docker"),
        host=environ.get("AP_HOST", "localhost"),
        port=environ.get("AP_PORT", "25432"),
        password=environ.get("AP_PASSWORD", "docker"),
    )
)
