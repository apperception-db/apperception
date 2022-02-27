import datetime

import psycopg2
from bounding_box import BoundingBox
from lens import PinholeLens
from new_util import create_camera, get_video, video_fetch_reformat
from pypika import Column, CustomFunction, Table
# https://github.com/kayak/pypika/issues/553
# workaround. because the normal Query will fail due to mobility db
from pypika.dialects import Query, SnowflakeQuery
from video_context import Camera
from video_util import add_recognized_objs, get_video_dimension, recognize

CAMERA_TABLE = "cameras"
TRAJ_TABLE = "item_general_trajectory"
BBOX_TABLE = "general_bbox"


class Database:
    def __init__(self, reset: bool = True):
        # should setup a postgres in docker first
        self.con = psycopg2.connect(
            dbname="mobilitydb", user="docker", host="localhost", port="25432", password="docker"
        )
        self.cur = self.con.cursor()

        if reset:
            # create camera table
            self._create_camera_table()

            # create bbox table
            self._create_general_bbox_table()

            # create traj table
            self._create_item_general_trajectory_table()

        # The start time of the database access object
        self.start_time = datetime.datetime(2021, 6, 8, 7, 10, 28)

    def _create_camera_table(self):
        # drop old
        q1 = SnowflakeQuery.drop_table(CAMERA_TABLE).if_exists()

        # create new
        q2 = SnowflakeQuery.create_table(CAMERA_TABLE).columns(
            Column("cameraId", "TEXT"),
            Column("worldId", "TEXT"),
            Column("ratio", "real"),
            Column("origin", "geometry"),
            Column("focalpoints", "geometry"),
            Column("fov", "INTEGER"),
            Column("skev_factor", "real"),
            Column("width", "integer"),
            Column("height", "integer"),
        )

        self.cur.execute(q1.get_sql())
        self.cur.execute(q2.get_sql())
        self.con.commit()

    def insert_cam(self, world_id: str, camera_node: Camera):
        cam = Table(CAMERA_TABLE)
        cam_id = camera_node.cam_id
        cam_ratio = camera_node.ratio
        lens = camera_node.lens

        if not isinstance(lens, PinholeLens):
            raise Exception("Only accept a camera with PinholeLens")

        focal_x = str(lens.focal_x)
        focal_y = str(lens.focal_y)
        cam_x, cam_y, cam_z = (
            str(lens.cam_origin[0]),
            str(lens.cam_origin[1]),
            str(lens.cam_origin[2]),
        )
        width, height = get_video_dimension(camera_node.video_file)

        q = SnowflakeQuery.into(cam).insert(
            cam_id,
            world_id,
            cam_ratio,
            f"POINT Z ({cam_x} {cam_y} {cam_z})",
            f"POINT({focal_x} {focal_y})",
            lens.fov,
            lens.alpha,
            width,
            height,
        )
        # print(q)
        self.cur.execute(q.get_sql())
        self.con.commit()

    def retrieve_cam(self, query: Query = None, world_id: str = ""):
        """
        Called when executing update commands (add_camera, add_objs ...etc)
        """

        return (
            query + self._select_cam_with_world_id(world_id)
            if query
            else self._select_cam_with_world_id(world_id)
        )  # UNION

    def _select_cam_with_world_id(self, world_id: str):
        """
        Select cams with certain world id
        """
        cam = Table(CAMERA_TABLE)
        q = SnowflakeQuery.from_(cam).select("*").where(cam.worldId == world_id)
        return q

    def filter_cam(self, query: Query, condition: str):
        """
        Called when executing filter commands (predicate, interval ...etc)
        """
        return SnowflakeQuery.from_(query).select("*").where(eval(condition))

    def get_cam(self, query: Query):
        """
        Execute sql command rapidly
        """

        # hack
        q = (
            "SELECT cameraId, ratio, ST_X(origin), ST_Y(origin), ST_Z(origin), ST_X(focalpoints), ST_Y(focalpoints), fov, skev_factor"
            + f" FROM ({query.get_sql()}) AS final"
        )

        # print(q)

        self.cur.execute(q)
        return self.cur.fetchall()

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

    def _create_general_bbox_table(self):
        # already created in create_or_insert_general_trajectory
        self.cur.execute("DROP TABLE IF EXISTS General_Bbox;")
        self.con.commit()

    def _create_item_general_trajectory_table(self):
        # already created in create_or_insert_general_trajectory
        self.cur.execute("DROP TABLE IF EXISTS Item_General_Trajectory;")
        self.con.commit()

    def insert_bbox_traj(self, world_id: str, camera_node: Camera, recognition_area: BoundingBox):
        video_file, algo, lens = camera_node.video_file, "Yolo", camera_node.lens
        tracking_results = recognize(
            video_file=video_file, recog_algo=algo, recognition_area=recognition_area
        )
        add_recognized_objs(self.con, lens, tracking_results, self.start_time, world_id)

    def retrieve_bbox(self, query: Query = None, world_id: str = ""):
        bbox = Table(BBOX_TABLE)
        q = SnowflakeQuery.from_(bbox).select("*").where(bbox.worldId == world_id)
        return query + q if query else q  # UNION

    def retrieve_traj(self, query: Query = None, world_id: str = ""):
        traj = Table(TRAJ_TABLE)
        q = SnowflakeQuery.from_(traj).select("*").where(traj.worldId == world_id)
        return query + q if query else q  # UNION

    def get_bbox(self, query: Query):
        self.cur.execute(query.get_sql())
        return self.cur.fetchall()

    def get_traj(self, query: Query):
        # hack
        query = (
            "SELECT asMFJSON(trajCentroids)::json->'coordinates'"
            + f" FROM ({query.get_sql()}) as final"
        )

        print("get_traj", query)
        self.cur.execute(query)
        return self.cur.fetchall()

    def get_traj_key(self, query: Query):
        q = SnowflakeQuery.from_(query).select("itemid")
        print("get_traj_key", q.get_sql())
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

    def get_video(self, query, cams):
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
        get_video(fetched_meta, cams, self.start_time)


if __name__ == "__main__":
    # Ingest the camera to the world
    c1 = create_camera("cam1", 120)
    c2 = create_camera("cam2", 150)

    db = Database()
    db.insert_bbox_traj(
        world_id="myworld", camera_node=c1, recognition_area=BoundingBox(0, 50, 50, 100)
    )
    db.insert_bbox_traj(
        world_id="myworld2", camera_node=c2, recognition_area=BoundingBox(0, 50, 50, 100)
    )

    q = db.retrieve_traj(world_id="myworld2")
    db.cur.execute(q.get_sql())
    res = db.cur.fetchall()
    print(res)

    # db.insert_cam("1", 5, "1")
    # db.insert_cam("2", 3, "2")
    # db.insert_cam("wid-1", c1)
    #
    # q = db.retrieve_cam(world_id="wid-1")
    # res = db.execute_get_query(q)
    # print(res)
    # # w1 = world()
    # q = ""
    # camera_node = Camera(cam_id, point, ratio, video_file, metadata_id, lens)
    # # w2 = w1.add_camera({cam_id: "1", size: 5, world_id: "1"})
    # q = db.concat_with(query = q, world_id = "1")
    # print(q)

    # # w3 = w2.predicate(cam.size < 4)
    # q = db.nest_from(query = q, condition = "query.size < 4")
    # print(q)

    # # w4 = w3.add_camera({cam_id: "2", size: 3, world_id: "2"})
    # q = db.concat_with(query = q, world_id = "2")
    # print(q)

    # # w5 = w4.predicate(cam.size < 6)
    # q = db.nest_from(query = q, condition = "query.size < 6")
    # print(q)

    # # w6 = w5.get_camera()
    # res = db.execute_get_query(query = q)

    # print(list(res))
