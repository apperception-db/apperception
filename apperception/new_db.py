import datetime
from types import FunctionType
from typing import Tuple

import psycopg2
from camera import Camera
from new_util import (add_recognized_objs, get_video, parse_predicate,
                      recognize, video_fetch_reformat)
from pypika import Column, CustomFunction, Table
# https://github.com/kayak/pypika/issues/553
# workaround. because the normal Query will fail due to mobility db
from pypika.dialects import Query, SnowflakeQuery
from pypika.functions import Cast
from scenic_util import fetch_camera as su_fetch_camera

CAMERA_TABLE = "Cameras"
TRAJ_TABLE = "Item_General_Trajectory"
BBOX_TABLE = "General_Bbox"


class Database:
    def __init__(self):
        # should setup a postgres in docker first
        self.con = psycopg2.connect(
            dbname="mobilitydb", user="docker", host="localhost", port="25432", password="docker"
        )
        self.cur = self.con.cursor()

        # The start time of the database access object
        self.start_time = datetime.datetime(2021, 6, 8, 7, 10, 28)

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
            Column("timestamp", "TEXT"),
            Column("cameraHeading", "real"),
            Column("egoHeading", "real"),
        )

        self.cur.execute(q1.get_sql())
        self.cur.execute(q2.get_sql())
        self.con.commit()

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

        self.cur.execute(q1.get_sql())
        self.cur.execute(q2)
        self.con.commit()

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
            itemHeadings real[],
            PRIMARY KEY (itemId)
        );
        """

        self.cur.execute(q1)
        self.cur.execute(q2)
        self.con.commit()

    def _create_index(self):
        self.cur.execute(
            """
            CREATE INDEX IF NOT EXISTS traj_idx
            ON Item_General_Trajectory
            USING GiST(trajCentroids);
        """
        )
        self.cur.execute(
            """
            CREATE INDEX IF NOT EXISTS item_idx
            ON General_Bbox(itemId);
        """
        )
        self.cur.execute(
            """
            CREATE INDEX IF NOT EXISTS traj_bbox_idx
            ON General_Bbox
            USING GiST(trajBbox);
        """
        )
        self.con.commit()

    def insert_cam(self, camera: Camera):
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
                '{config.timestamp}',
                {config.cameraHeading},
                {config.egoHeading}
            )"""
            for config in camera.configs
        ]

        self.cur.execute(
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
        self.con.commit()

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

    def filter_cam(self, query: Query, condition: str):
        """
        Called when executing filter commands (predicate, interval ...etc)
        """
        return SnowflakeQuery.from_(query).select("*").where(eval(condition))

    def predicate(self, query: Query, func: FunctionType):
        condition = parse_predicate(query, func)
        return SnowflakeQuery.from_(query).select("*").where(eval(condition))

    def get_cam(self, query: Query):
        """
        Execute sql command rapidly
        """

        # hack
        q = (
            "SELECT cameraID, frameId, frameNum, fileName, cameraTranslation, cameraRotation, cameraIntrinsic, egoTranslation, egoRotation, timestamp, cameraHeading, egoHeading"
            + f" FROM ({query.get_sql()}) AS final"
        )

        # print(q)

        self.cur.execute(q)
        return self.cur.fetchall()

    def fetch_camera(self, scene_name: str, frame_num: int):
        # TODO: more specific return type: Any, List[...]
        return su_fetch_camera(self.con, scene_name, frame_num)

    def get_len(self, query: Query):
        """
        Execute sql command rapidly
        """

        # hack
        q = (
            "SELECT ratio, ST_X(origin), ST_Y(origin), ST_Z(origin), fov, skev_factor"
            + f" FROM ({query.get_sql()}) AS final"
        )

        self.cur.execute(q)
        return self.cur.fetchall()

    def insert_bbox_traj(self, camera: Camera, annotation):
        tracking_results = recognize(camera.configs, annotation)
        add_recognized_objs(self.con, tracking_results, self.start_time, camera.id)

    def retrieve_bbox(self, query: Query = None, camera_id: str = ""):
        bbox = Table(BBOX_TABLE)
        q = SnowflakeQuery.from_(bbox).select("*").where(bbox.cameraId == camera_id)
        return query + q if query else q  # UNION

    def retrieve_traj(self, query: Query = None, camera_id: str = ""):
        traj = Table(TRAJ_TABLE)
        q = SnowflakeQuery.from_(traj).select("*").where(traj.cameraId == camera_id)
        return query + q if query else q  # UNION

    def get_bbox(self, query: Query):
        self.cur.execute(query.get_sql())
        return self.cur.fetchall()

    def get_traj(self, query: Query):
        # hack
        query = (
            "SELECT asMFJSON(trajCentroids)::json->'sequences'"
            + f" FROM ({query.get_sql()}) as final"
        )

        print("get_traj")  # print("get_traj", query)
        self.cur.execute(query)
        return self.cur.fetchall()

    def get_traj_key(self, query: Query):
        q = SnowflakeQuery.from_(query).select("itemid")
        print("get_traj_key")  # print("get_traj_key", q.get_sql())
        self.cur.execute(q.get_sql())
        return self.cur.fetchall()

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
        self.cur.execute(q.get_sql())
        return self.cur.fetchall()

    def get_time(self, query: Query):
        Tmin = CustomFunction("Tmin", ["stbox"])
        q = SnowflakeQuery.from_(query).select(Tmin(query.trajBbox))
        self.cur.execute(q.get_sql())
        return self.cur.fetchall()

    def get_distance(self, query: Query, start: str, end: str):
        atPeriodSet = CustomFunction("atPeriodSet", ["centroids", "param"])
        cumulativeLength = CustomFunction("cumulativeLength", ["input"])
        q = SnowflakeQuery.from_(query).select(
            cumulativeLength(atPeriodSet(query.trajCentroids, "{[%s, %s)}" % (start, end)))
        )

        self.cur.execute(q.get_sql())
        return self.cur.fetchall()

    def get_speed(self, query, start, end):
        atPeriodSet = CustomFunction("atPeriodSet", ["centroids", "param"])
        speed = CustomFunction("speed", ["input"])

        q = SnowflakeQuery.from_(query).select(
            speed(atPeriodSet(query.trajCentroids, "{[%s, %s)}" % (start, end)))
        )

        self.cur.execute(q.get_sql())
        return self.cur.fetchall()

    def filter_traj_type(self, query: Query, object_type: str):
        return SnowflakeQuery.from_(query).select("*").where(query.objecttype == object_type)

    def filter_traj_heading(self, query: Query, lessThan=float("inf"), greaterThan=float("-inf")):
        return (
            SnowflakeQuery.from_(query)
            .select("*")
            .where(query.heading <= lessThan)
            .where(query.heading >= greaterThan)
        )

    def filter_distance_to_type(self, query: Query, distance: float, type: str):
        # TODO: Implement Types
        cameras = Table(CAMERA_TABLE)
        getX = CustomFunction("getX", ["tgeompoint"])
        getY = CustomFunction("getY", ["tgeompoint"])
        valueAtTimestamp = CustomFunction("valueAtTimestamp", ["tfloat", "timestamptz"])

        ST_X = CustomFunction("ST_X", ["geometry"])
        ST_Y = CustomFunction("ST_Y", ["geometry"])
        ST_Centroid = CustomFunction("ST_Centroid", ["geometry"])
        SQRT = CustomFunction("SQRT", ["number"])
        POWER = CustomFunction("POWER", ["number", "number"])
        camera_time = Cast(self.start_time, "timestamptz") + cameras.frameNum * Cast(
            "1 second", "interval"
        )
        subtract_x = valueAtTimestamp(getX(query.trajCentroids), camera_time) - ST_X(
            ST_Centroid(cameras.egoTranslation)
        )
        subtract_y = valueAtTimestamp(getY(query.trajCentroids), camera_time) - ST_Y(
            ST_Centroid(cameras.egoTranslation)
        )
        subtract_mag = SQRT(POWER(subtract_x, 2) + POWER(subtract_y, 2))
        q = (
            SnowflakeQuery.from_(query)
            .join(cameras)
            .cross()
            .select(query.star)
            .distinct()
            .where(subtract_mag <= distance)
        )
        return q

    def filter_relative_to_type(
        self,
        query: Query,
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        z_range: Tuple[float, float],
        type: str,
    ):
        # TODO: Make also work with objects of other types
        # TODO: Fix Up Local Coordinate Frame Stuff
        cameras = Table(CAMERA_TABLE)
        getX = CustomFunction("getX", ["tgeompoint"])
        getY = CustomFunction("getY", ["tgeompoint"])
        valueAtTimestamp = CustomFunction("valueAtTimestamp", ["tfloat", "timestamptz"])

        ST_X = CustomFunction("ST_X", ["geometry"])
        ST_Y = CustomFunction("ST_Y", ["geometry"])
        ST_Centroid = CustomFunction("ST_Centroid", ["geometry"])
        SQRT = CustomFunction("SQRT", ["number"])
        COS = CustomFunction("COS", ["number"])
        SIN = CustomFunction("SIN", ["number"])
        ATAN2 = CustomFunction("ATAN2", ["number", "number"])
        POWER = CustomFunction("POWER", ["number", "number"])
        PI = CustomFunction("PI", [])
        camera_time = Cast(self.start_time, "timestamptz") + cameras.frameNum * Cast(
            "1 second", "interval"
        )

        subtract_x = valueAtTimestamp(getX(query.trajCentroids), camera_time) - ST_X(
            ST_Centroid(cameras.egoTranslation)
        )
        subtract_y = valueAtTimestamp(getY(query.trajCentroids), camera_time) - ST_Y(
            ST_Centroid(cameras.egoTranslation)
        )
        subtract_mag = SQRT(POWER(subtract_x, 2) + POWER(subtract_y, 2))
        # ST_Z = CustomFunction("ST_Z", ["geometry"])
        ST_Centroid = CustomFunction("ST_Centroid", ["geometry"])

        q = (
            SnowflakeQuery.from_(query)
            .join(cameras)
            .cross()
            .select(query.star)
            .distinct()
            .where(
                x_range[0]
                <= (
                    subtract_mag
                    * COS(PI() * cameras.egoHeading / 180 + ATAN2(subtract_y, subtract_x))
                )
            )
            .where(
                (
                    subtract_mag
                    * COS(PI() * cameras.egoHeading / 180 + ATAN2(subtract_y, subtract_x))
                )
                <= x_range[1]
            )
            .where(
                y_range[0]
                <= (
                    subtract_mag
                    * SIN(PI() * cameras.egoHeading / 180 + ATAN2(subtract_y, subtract_x))
                )
            )
            .where(
                (
                    subtract_mag
                    * SIN(PI() * cameras.egoHeading / 180 + ATAN2(subtract_y, subtract_x))
                )
                <= y_range[1]
            )
        )

        # q2 = (
        #     SnowflakeQuery.from_(query)
        #     .join(cameras)
        #     .cross()
        #     .select(query.itemId, cameras.filename, cameras.heading, (subtract_mag * COS(PI()*cameras.heading/180 + ATAN2(subtract_y, subtract_x))), (subtract_mag * SIN(PI()*cameras.heading/180 + ATAN2(subtract_y, subtract_x))))
        #     .where(
        #         x_range[0]
        #         <= (
        #             subtract_mag * COS(PI() * cameras.heading / 180 + ATAN2(subtract_y, subtract_x))
        #         )
        #     )
        #     .where(
        #         (subtract_mag * COS(PI() * cameras.heading / 180 + ATAN2(subtract_y, subtract_x)))
        #         <= x_range[1]
        #     )
        #     .where(
        #         y_range[0]
        #         <= (
        #             subtract_mag * SIN(PI() * cameras.heading / 180 + ATAN2(subtract_y, subtract_x))
        #         )
        #     )
        #     .where(
        #         (subtract_mag * SIN(PI() * cameras.heading / 180 + ATAN2(subtract_y, subtract_x)))
        #         <= y_range[1]
        #     )
        # )
        # print("yeeee boy")
        # self.cur.execute(q2.get_sql())
        # [print(x) for x in self.cur.fetchall() if "/CAM_FRONT/" in x[1]]
        print(str(q))

        return q

    def filter_traj_volume(self, query: Query, volume: str):
        overlap = CustomFunction("overlap", ["bbox1", "bbox2"])
        return SnowflakeQuery.from_(query).select("*").where(overlap(query.largestBbox, volume))

    def interval(self, query, start, end):
        # https://pypika.readthedocs.io/en/latest/4_extending.html
        Tmin = CustomFunction("Tmin", ["stbox"])
        Tmax = CustomFunction("Tmax", ["stbox"])
        return (
            SnowflakeQuery.from_(query)
            .select("*")
            .where((start <= Tmin(query.trajBbox)) & (Tmax(query.trajBbox) < end))
        )

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

        self.cur.execute(query.get_sql())
        fetched_meta = self.cur.fetchall()
        fetched_meta = video_fetch_reformat(fetched_meta)
        get_video(fetched_meta, cams, self.start_time, boxed)

    def get_heading_from_a_point(self, x, y):
        # query = f"SELECT heading FROM Segment WHERE elementid IN (SELECT Polygon.elementid AS id FROM Polygon, ST_Point({x}, {y}) AS point WHERE ST_Contains(elementPolygon, point)='t');"
        query = f"""
            select heading from segment, st_point({x}, {y}) as point, st_distance(st_makeline(startPoint, endPoint), point) as dis
            order by dis asc
            limit 1;
        """
        self.cur.execute(query)
        return self.cur.fetchall()


setattr(Database.insert_bbox_traj, "comparators", {"annotation": lambda df: df[0].equals(df[1])})

if __name__ == "__main__":
    x, y = 1317, 1463
    db = Database()
    print(db.get_heading_from_a_point(x, y))
