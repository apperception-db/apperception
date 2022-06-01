import ast
import inspect
from datetime import datetime
from typing import TYPE_CHECKING, Callable, List, Optional, Union

import psycopg2
from pypika import Column, CustomFunction, Table
# https://github.com/kayak/pypika/issues/553
# workaround. because the normal Query will fail due to mobility db
from pypika.dialects import Query, SnowflakeQuery

import apperception.scenic_util as su
from apperception.data_types import QueryType, Trajectory
from apperception.utils import (add_recognized_objects, fn_to_sql,
                                overlay_bboxes, query_to_str, recognize,
                                reformat_bbox_trajectories)

if TYPE_CHECKING:
    from psycopg2 import connection as Connection
    from psycopg2 import cursor as Cursor

    from .data_types import Camera
    from .new_world import World

CAMERA_TABLE = "Cameras"
TRAJ_TABLE = "Item_General_Trajectory"
BBOX_TABLE = "General_Bbox"


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

    def reset(self):
        self._create_camera_table()
        self._create_item_general_trajectory_table()
        self._create_general_bbox_table()
        self._create_index()

    def _create_camera_table(self):
        # drop old
        q1 = SnowflakeQuery.drop_table(CAMERA_TABLE).if_exists()

        # create new
        q2 = SnowflakeQuery.create_table(CAMERA_TABLE).columns(
            Column("cameraId", "TEXT"),
            Column("frameId", "TEXT"),
            Column("frameNum", "Int"),
            Column("fileName", "TEXT"),
            Column("cameraTranslation", "geometry"),
            Column("cameraRotation", "real[4]"),
            Column("cameraIntrinsic", "real[3][3]"),
            Column("egoTranslation", "geometry"),
            Column("egoRotation", "real[4]"),
            Column("timestamp", "timestamptz"),
            Column("cameraHeading", "real"),
            Column("egoHeading", "real"),
        )
        self.cursor.execute(q1.get_sql())
        self.cursor.execute(q2.get_sql())
        self.connection.commit()

    def _create_general_bbox_table(self):
        # drop old
        q1 = SnowflakeQuery.drop_table(BBOX_TABLE).if_exists()

        # create new
        q2 = """
        CREATE TABLE General_Bbox(
            itemId TEXT,
            cameraId TEXT,
            trajBbox stbox,
            FOREIGN KEY(itemId)
                REFERENCES Item_General_Trajectory(itemId)
        );
        """

        self.cursor.execute(q1.get_sql())
        self.cursor.execute(q2)
        self.connection.commit()

    def _create_item_general_trajectory_table(self):
        # drop old
        q1 = "DROP TABLE IF EXISTS Item_General_Trajectory CASCADE;"

        # create new
        q2 = """
        CREATE TABLE Item_General_Trajectory(
            itemId TEXT,
            cameraId TEXT,
            objectType TEXT,
            color TEXT,
            trajCentroids tgeompoint,
            largestBbox stbox,
            itemHeadings tfloat,
            PRIMARY KEY (itemId)
        );
        """

        self.cursor.execute(q1)
        self.cursor.execute(q2)
        self.connection.commit()

    def _create_index(self):
        self.cursor.execute(
            """
            CREATE INDEX ON Cameras (cameraId);
            """
        )

        self.cursor.execute(
            """
            CREATE INDEX ON Cameras (timestamp);
            """
        )

        self.cursor.execute(
            """
            CREATE INDEX ON Item_General_Trajectory (itemId);
            """
        )

        self.cursor.execute(
            """
            CREATE INDEX ON Item_General_Trajectory (cameraId);
            """
        )

        self.cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS traj_idx
            ON Item_General_Trajectory
            USING GiST(trajCentroids);
        """
        )
        self.cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS item_idx
            ON General_Bbox(itemId);
        """
        )
        self.cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS traj_bbox_idx
            ON General_Bbox
            USING GiST(trajBbox);
        """
        )
        self.connection.commit()

    def insert_cam(self, camera: "Camera"):
        values = [
            f"""(
                '{camera.id}',
                '{config.frame_id}',
                {config.frame_num},
                '{config.filename}',
                'POINT Z ({' '.join(map(str, config.camera_translation))})',
                ARRAY{config.camera_rotation},
                ARRAY{config.camera_intrinsic},
                'POINT Z ({' '.join(map(str, config.ego_translation))})',
                ARRAY{config.ego_rotation},
                '{datetime.fromtimestamp(float(config.timestamp)/1000000.0)}',
                {config.cameraHeading},
                {config.egoHeading}
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
                egoHeading
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

    def filter(self, query: Query, predicate: Union[str, Callable]):
        if isinstance(predicate, str):
            body = ast.parse(predicate).body[0]
            if isinstance(body, ast.FunctionDef):
                args = [arg.arg for arg in body.args.args]
            elif isinstance(body, ast.Expr):
                value = body.value
                if isinstance(value, ast.Lambda):
                    args = [arg.arg for arg in value.args.args]
                else:
                    raise Exception("Predicate is not a function")
            else:
                raise Exception("Predicate is not a function")
        else:
            args = inspect.getfullargspec(predicate).args

        tables = []
        tables_sql = []
        table_idx = 0
        found_camera = False
        found_road = False
        for arg in args:
            if arg in ["c", "cam", "camera"]:
                if found_camera:
                    raise Exception("Only allow one camera parameter")
                tables_sql.append("Cameras")
                found_camera = True
            elif arg in ["r", "road"]:
                if found_road:
                    raise Exception("Only allow one road parameter")
                # TODO: Road is not a real DB table name
                tables_sql.append("Road")
                found_road = True
            else:
                # TODO: table name should depend on world's id
                table_name = f"table_{table_idx}"
                tables.append(table_name)
                tables_sql.append(table_name)
                table_idx += 1

        predicate_sql = fn_to_sql(predicate, tables_sql)
        query_str = query_to_str(query)
        joins = [
            f"JOIN ({query_str}) as {table} ON {table}.cameraId = {tables[0]}.cameraId"
            for table in tables[1:]
        ]

        return f"""
        SELECT DISTINCT {tables[0]}.*
        FROM ({query_str}) as {tables[0]}
        {" ".join(joins)}
        {f"JOIN Cameras ON Cameras.cameraId = {tables[0]}.cameraId" if found_camera else ""}
        WHERE {predicate_sql}
        """

    def exclude(self, query: Query, world: "World"):
        return f"""
        SELECT *
        FROM ({query_to_str(query)}) as __query__
        EXCEPT
        SELECT *
        FROM ({world._execute_from_root(QueryType.TRAJ)}) as __except__
        """

    def union(self, query: Query, world: "World"):
        return f"""
        SELECT *
        FROM ({query_to_str(query)}) as __query__
        UNION
        SELECT *
        FROM ({world._execute_from_root(QueryType.TRAJ)}) as __union__
        """

    def intersect(self, query: Query, world: "World"):
        return f"""
        SELECT *
        FROM ({query_to_str(query)}) as __query__
        INTERSECT
        SELECT *
        FROM ({world._execute_from_root(QueryType.TRAJ)}) as __intersect__
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

        self.cursor.execute(q)
        return self.cursor.fetchall()

    def fetch_camera(self, scene_name: str, frame_timestamp: List[str]):
        return su.fetch_camera(self.connection, scene_name, frame_timestamp)

    def fetch_camera_framenum(self, scene_name: str, frame_num: List[int]):
        return su.fetch_camera_framenum(self.connection, scene_name, frame_num)

    def timestamp_to_framenum(self, scene_name: str, timestamps: List[str]):
        return su.timestamp_to_framenum(self.connection, scene_name, timestamps)

    def get_len(self, query: Query):
        """
        Execute sql command rapidly
        """

        # hack
        q = (
            "SELECT ratio, ST_X(origin), ST_Y(origin), ST_Z(origin), fov, skev_factor"
            + f" FROM ({query.get_sql()}) AS final"
        )

        self.cursor.execute(q)
        return self.cursor.fetchall()

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

    def road_direction(self, x: float, y: float):
        q = f"SELECT roadDirection({x}, {y});"
        self.cursor.execute(q)
        return self.cursor.fetchall()

    def road_coords(self, x: float, y: float):
        q = f"SELECT roadCoords({x}, {y});"
        self.cursor.execute(q)
        return self.cursor.fetchall()

    def get_bbox(self, query: Query):
        self.cursor.execute(query.get_sql())
        return self.cursor.fetchall()

    def get_traj(self, query: Query) -> List[List[Trajectory]]:
        # hack
        query = f"""
        SELECT asMFJSON(trajCentroids)::json->'sequences'
        FROM ({query_to_str(query)}) as final
        """

        print("get_traj", query)
        self.cursor.execute(query)
        trajectories = self.cursor.fetchall()
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
        query = f"""
        SELECT itemId FROM ({query_to_str(query)}) as final
        """

        print("get_traj_key", query)
        self.cursor.execute(query)
        return self.cursor.fetchall()

    def get_traj_attr(self, query: Query, attr: str):
        query = f"""
        SELECT {attr} FROM ({query_to_str(query)}) as final
        """

        print("get_traj_attr:", attr, query)
        self.cursor.execute(query)
        return self.cursor.fetchall()

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
        self.cursor.execute(q.get_sql())
        return self.cursor.fetchall()

    def get_time(self, query: Query):
        Tmin = CustomFunction("Tmin", ["stbox"])
        q = SnowflakeQuery.from_(query).select(Tmin(query.trajBbox))
        self.cursor.execute(q.get_sql())
        return self.cursor.fetchall()

    def get_distance(self, query: Query, start: str, end: str):
        atPeriodSet = CustomFunction("atPeriodSet", ["centroids", "param"])
        cumulativeLength = CustomFunction("cumulativeLength", ["input"])
        q = SnowflakeQuery.from_(query).select(
            cumulativeLength(atPeriodSet(query.trajCentroids, "{[%s, %s)}" % (start, end)))
        )

        self.cursor.execute(q.get_sql())
        return self.cursor.fetchall()

    def get_speed(self, query, start, end):
        atPeriodSet = CustomFunction("atPeriodSet", ["centroids", "param"])
        speed = CustomFunction("speed", ["input"])

        q = SnowflakeQuery.from_(query).select(
            speed(atPeriodSet(query.trajCentroids, "{[%s, %s)}" % (start, end)))
        )

        self.cursor.execute(q.get_sql())
        return self.cursor.fetchall()

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

        self.cursor.execute(query.get_sql())
        fetched_meta = self.cursor.fetchall()
        fetched_meta = reformat_bbox_trajectories(fetched_meta)
        overlay_bboxes(fetched_meta, cams, boxed)


database = Database()
