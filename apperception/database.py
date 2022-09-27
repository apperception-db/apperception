from datetime import datetime
from os import environ
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple

import pandas as pd
import psycopg2
import psycopg2.errors
from pypika import CustomFunction, Table
# https://github.com/kayak/pypika/issues/553
# workaround. because the normal Query will fail due to mobility db
from pypika.dialects import Query, SnowflakeQuery

from apperception.data_types import Trajectory
from apperception.predicate import (FindAllTablesVisitor, GenSqlVisitor,
                                    MapTablesTransformer, normalize)
from apperception.utils import (add_recognized_objects, fetch_camera,
                                fetch_camera_framenum, overlay_bboxes,
                                query_to_str, recognize,
                                reformat_bbox_trajectories,
                                timestamp_to_framenum)

if TYPE_CHECKING:
    from psycopg2 import connection as Connection
    from psycopg2 import cursor as Cursor

    from .data_types import Camera
    from .predicate import PredicateNode
    from .world import World

CAMERA_TABLE = "Cameras"
TRAJ_TABLE = "Item_General_Trajectory"
BBOX_TABLE = "General_Bbox"

CAMERA_COLUMNS: List[Tuple[str, str]] = [
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
    ("cameraTranslationAbs", "geometry"),
]

TRAJECTORY_COLUMNS: List[Tuple[str, str]] = [
    ("itemId", "TEXT"),
    ("cameraId", "TEXT"),
    ("objectType", "TEXT"),
    ("color", "TEXT"),
    ("trajCentroids", "tgeompoint"),
    ("translations", "tgeompoint"),  # [(x,y,z)@today, (x2, y2,z2)@tomorrow, (x2, y2,z2)@nextweek]
    ("largestBbox", "stbox"),
    ("itemHeadings", "tfloat"),
    # ("period", "period") [today, nextweek]
]

BBOX_COLUMNS: List[Tuple[str, str]] = [
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

    def __init__(self, connection: Optional["Connection"] = None):
        # should setup a postgres in docker first
        if connection is None:
            self.connection = psycopg2.connect(
                dbname="mobilitydb",
                user="docker",
                host="localhost",
                port="25432",
                password="docker",
            )
        else:
            self.connection = connection
        self.cursor = self.connection.cursor()

    def reset(self, commit=True):
        self._create_camera_table(False)
        self._create_item_general_trajectory_table(False)
        self._create_general_bbox_table(False)
        self._create_index(False)
        self._commit(commit)

    def _create_camera_table(self, commit=True):
        self.cursor.execute("DROP TABLE IF EXISTS Cameras CASCADE;")
        self.cursor.execute(f"CREATE TABLE Cameras ({columns(_schema, CAMERA_COLUMNS)})")
        self._commit(commit)

    def _create_general_bbox_table(self, commit=True):
        self.cursor.execute("DROP TABLE IF EXISTS General_Bbox CASCADE;")
        self.cursor.execute(
            f"""
            CREATE TABLE General_Bbox (
                {columns(_schema, BBOX_COLUMNS)},
                FOREIGN KEY(itemId) REFERENCES Item_General_Trajectory(itemId),
                PRIMARY KEY (itemId, timestamp)
            )
            """
        )
        self._commit(commit)

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
        self.cursor.execute("CREATE INDEX IF NOT EXISTS item_idx ON General_Bbox(itemId);")
        self.cursor.execute(
            "CREATE INDEX IF NOT EXISTS traj_bbox_idx ON General_Bbox USING GiST(trajBbox);"
        )
        self.cursor.execute(
            "CREATE INDEX IF NOT EXISTS item_id_timestampx ON General_Bbox(itemId, timestamp);"
        )
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

    def _insert_into_general_bbox(self, value: tuple, commit=True):
        self.cursor.execute(
            f"INSERT INTO General_Bbox ({columns(_name, BBOX_COLUMNS)}) VALUES ({place_holder(len(BBOX_COLUMNS))})",
            tuple(value),
        )
        self._commit(commit)

    def _commit(self, commit=True):
        if commit:
            self.connection.commit()

    def _execute_query(self, query: str) -> List[tuple]:
        try:
            self.cursor.execute(query)
            for notice in self.cursor.connection.notices:
                print(notice)
            if self.cursor.pgresult_ptr is not None:
                return self.cursor.fetchall()
            else:
                return []
        except psycopg2.errors.DatabaseError as error:
            self.connection.rollback()
            raise error

    def _execute_update(self, query: str, commit: bool = True) -> None:
        try:
            self.cursor.execute(query)
            self._commit(commit)
        except psycopg2.errors.DatabaseError as error:
            self.connection.rollback()
            raise error

    def insert_cam                          (self, camera: "Camera"):
        values = [
            f"""(
                '{camera.id}',
                '{config.frame_id}',
                {config.frame_num},
                '{config.filename}',
                'POINT Z ({' '.join(map(str, config.camera_translation))})',
                ARRAY{config.camera_rotation}::real[],
                ARRAY{config.camera_intrinsic}::real[][],
                'POINT Z ({' '.join(map(str, config.ego_translation))})',
                ARRAY{config.ego_rotation}::real[],
                '{datetime.fromtimestamp(float(config.timestamp)/1000000.0)}',
                {config.cameraHeading},
                {config.egoHeading},
                'POINT Z ({' '.join(map(str, config.camera_translation_abs))})'
            )"""
            for config in camera.configs
        ]

        self.cursor.execute(
            f"""
            INSERT INTO Cameras (
                cameraId,
                frameId,
                frameNum,
                fileName,
                cameraTranslation,
                cameraRotation,
                cameraIntrinsic,
                egoTranslation,
                egoRotation,
                timestamp,
                cameraHeading,
                egoHeading,
                cameraTranslationAbs
            )
            VALUES {','.join(values)};
            """
        )

        print("New camera inserted successfully.........")
        self.connection.commit()

    def retrieve_cam(self, query: Query = None, camera_id: str = ""):
        """
        Called when executing update commands (add_camera, add_objs ...etc)
        """

        return (
            query + self._select_cam_with_camera_id(camera_id)
            if query
            else self._select_cam_with_camera_id(camera_id)
        )  # UNION

    def _select_cam_with_camera_id(self, camera_id: str):
        """
        Select cams with certain world id
        """
        cam = Table(CAMERA_TABLE)
        q = SnowflakeQuery.from_(cam).select("*").where(cam.cameraId == camera_id)
        return q

    def filter(self, query: Query, predicate: "PredicateNode"):
        tables, camera = FindAllTablesVisitor()(predicate)
        tables = sorted(tables)
        mapping = {t: i for i, t in enumerate(tables)}
        predicate = normalize(predicate)
        predicate = MapTablesTransformer(mapping)(predicate)
        query_str = query_to_str(query)
        joins = [f"JOIN ({query_str}) as t{i} USING (cameraId)" for i in range(1, len(tables))]

        return f"""
        SELECT DISTINCT *
        FROM ({query_str}) as t0
        {" ".join(joins)}
        {f"JOIN Cameras USING (cameraId)" if camera else ""}
        WHERE {GenSqlVisitor()(predicate)}
        """

    def exclude(self, query: Query, world: "World"):
        return f"""
        SELECT *
        FROM ({query_to_str(query)}) as __query__
        EXCEPT
        SELECT *
        FROM ({world._execute_from_root()}) as __except__
        """

    def union(self, query: Query, world: "World"):
        return f"""
        SELECT *
        FROM ({query_to_str(query)}) as __query__
        UNION
        SELECT *
        FROM ({world._execute_from_root()}) as __union__
        """

    def intersect(self, query: Query, world: "World"):
        return f"""
        SELECT *
        FROM ({query_to_str(query)}) as __query__
        INTERSECT
        SELECT *
        FROM ({world._execute_from_root()}) as __intersect__
        """

    def get_cam(self, query: Query):
        """
        Execute sql command rapidly
        """

        # hack
        q = (
            "SELECT cameraID, frameId, frameNum, fileName, cameraTranslation, cameraRotation, cameraIntrinsic, egoTranslation, egoRotation, timestamp, cameraHeading, egoHeading"
            + f" FROM ({query.get_sql()}) AS final"
        )
        return self._execute_query(q)

    def fetch_camera(self, scene_name: str, frame_timestamp: List[str]):
        return fetch_camera(self.connection, scene_name, frame_timestamp)

    def fetch_camera_framenum(self, scene_name: str, frame_num: List[int]):
        return fetch_camera_framenum(self.connection, scene_name, frame_num)

    def timestamp_to_framenum(self, scene_name: str, timestamps: List[str]):
        return timestamp_to_framenum(self.connection, scene_name, timestamps)

    def get_len(self, query: Query):
        """
        Execute sql command rapidly
        """

        # hack
        q = (
            "SELECT ratio, ST_X(origin), ST_Y(origin), ST_Z(origin), fov, skev_factor"
            + f" FROM ({query.get_sql()}) AS final"
        )
        return self._execute_query(q)

    def insert_bbox_traj(self, camera: "Camera", annotation):
        tracking_results = recognize(camera.configs, annotation)
        add_recognized_objects(self.connection, tracking_results, camera.id)

    def retrieve_bbox(self, query: Query = None, camera_id: str = ""):
        bbox = Table(BBOX_TABLE)
        q = SnowflakeQuery.from_(bbox).select("*").where(bbox.cameraId == camera_id)
        return query + q if query else q  # UNION

    def retrieve_traj(self, query: Query = None, camera_id: str = ""):
        traj = Table(TRAJ_TABLE)
        q = SnowflakeQuery.from_(traj).select("*").where(traj.cameraId == camera_id)
        return query + q if query else q  # UNION

    def road_direction(self, x: float, y: float, default_dir: float):
        return self._execute_query(f"SELECT roadDirection({x}, {y}, {default_dir});")

    def road_coords(self, x: float, y: float):
        return self._execute_query(f"SELECT roadCoords({x}, {y});")

    def select_all(self, query: "Query") -> List[tuple]:
        _query = query_to_str(query)
        print("select_all:", _query)
        return self._execute_query(_query)

    def get_traj(self, query: Query) -> List[List[Trajectory]]:
        # hack
        query = f"""
        SELECT asMFJSON(trajCentroids)::json->'sequences'
        FROM ({query_to_str(query)}) as final
        """

        print("get_traj", query)
        trajectories = self._execute_query(query)
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

    def get_traj_key(self, query: Query):
        _query = f"""
        SELECT itemId FROM ({query_to_str(query)}) as final
        """

        print("get_traj_key", _query)
        return self._execute_query(_query)

    def get_id_time_camId_filename(self, query: Query, num_joined_tables: int):
        itemId = ",".join([f"t{i}.itemId" for i in range(num_joined_tables)])
        timestamp = "cameras.timestamp"
        camId = "cameras.cameraId"
        filename = "cameras.filename"
        _query = query_to_str(query).replace(
            "SELECT DISTINCT *", f"SELECT {itemId}, {timestamp}, {camId}, {filename}", 1
        )

        print("get_id_time_camId_filename", _query)
        return self._execute_query(_query)

    def get_traj_attr(self, query: Query, attr: str):
        _query = f"""
        SELECT {attr} FROM ({query_to_str(query)}) as final
        """

        print("get_traj_attr:", attr, _query)
        return self._execute_query(_query)

    def get_bbox_geo(self, query: Query):
        Xmin = CustomFunction("Xmin", ["stbox"])
        Ymin = CustomFunction("Ymin", ["stbox"])
        Zmin = CustomFunction("Zmin", ["stbox"])
        Xmax = CustomFunction("Xmax", ["stbox"])
        Ymax = CustomFunction("Ymax", ["stbox"])
        Zmax = CustomFunction("Zmax", ["stbox"])

        q = SnowflakeQuery.from_(query).select(
            Xmin(query.trajBbox),
            Ymin(query.trajBbox),
            Zmin(query.trajBbox),
            Xmax(query.trajBbox),
            Ymax(query.trajBbox),
            Zmax(query.trajBbox),
        )
        return self._execute_query(q.get_sql())

    def get_time(self, query: Query):
        Tmin = CustomFunction("Tmin", ["stbox"])
        q = SnowflakeQuery.from_(query).select(Tmin(query.trajBbox))
        return self._execute_query(q.get_sql())

    def get_distance(self, query: Query, start: str, end: str):
        atPeriodSet = CustomFunction("atPeriodSet", ["centroids", "param"])
        cumulativeLength = CustomFunction("cumulativeLength", ["input"])
        q = SnowflakeQuery.from_(query).select(
            cumulativeLength(atPeriodSet(query.trajCentroids, "{[%s, %s)}" % (start, end)))
        )

        return self._execute_query(q.get_sql())

    def get_speed(self, query, start, end):
        atPeriodSet = CustomFunction("atPeriodSet", ["centroids", "param"])
        speed = CustomFunction("speed", ["input"])

        q = SnowflakeQuery.from_(query).select(
            speed(atPeriodSet(query.trajCentroids, "{[%s, %s)}" % (start, end)))
        )

        return self._execute_query(q.get_sql())

    def get_video(self, query, cams, boxed):
        bbox = Table(BBOX_TABLE)
        Xmin = CustomFunction("Xmin", ["stbox"])
        Ymin = CustomFunction("Ymin", ["stbox"])
        Zmin = CustomFunction("Zmin", ["stbox"])
        Xmax = CustomFunction("Xmax", ["stbox"])
        Ymax = CustomFunction("Ymax", ["stbox"])
        Zmax = CustomFunction("Zmax", ["stbox"])
        Tmin = CustomFunction("Tmin", ["stbox"])

        query = (
            SnowflakeQuery.from_(query)
            .inner_join(bbox)
            .using("itemid")
            .select(
                query.itemid,
                Xmin(bbox.trajBbox),
                Ymin(bbox.trajBbox),
                Zmin(bbox.trajBbox),
                Xmax(bbox.trajBbox),
                Ymax(bbox.trajBbox),
                Zmax(bbox.trajBbox),
                Tmin(bbox.trajBbox),
            )
        )

        fetched_meta = self._execute_query(query.get_sql())
        _fetched_meta = reformat_bbox_trajectories(fetched_meta)
        overlay_bboxes(_fetched_meta, cams, boxed)

    def sql(self, query: str) -> pd.DataFrame:
        return pd.DataFrame(
            self._execute_query(query), columns=[d.name for d in self.cursor.description]
        )


database = Database(
    psycopg2.connect(
        dbname=environ.get("AP_DB", "mobilitydb"),
        user=environ.get("AP_USER", "docker"),
        host=environ.get("AP_HOST", "localhost"),
        port=environ.get("AP_PORT", "25432"),
        password=environ.get("AP_PASSWORD", "docker"),
    )
)
