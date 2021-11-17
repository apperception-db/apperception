import sqlite3
import psycopg2
import lens
import point
from pypika import Table, Field, Column
from video_context import Camera
from video_util import get_video_dimension
# https://github.com/kayak/pypika/issues/553
# workaround. because the normal Query will fail due to mobility db
from pypika.dialects import SnowflakeQuery as Query


CAMERA_TABLE = "cameras"

class Database:
    def __init__(self):
        # should setup a postgres in docker first
        self.con = psycopg2.connect(dbname="mobilitydb", user="docker", host="localhost", port="25432",
                                    password="docker")
        self.cur = self.con.cursor()

        # create camera table
        self._create_camera_table()

    def _create_camera_table(self):
        # drop old
        q1 = (
            Query
                .drop_table(CAMERA_TABLE)
                .if_exists()
        )

        # create new
        q2 = (
            Query
                .create_table(CAMERA_TABLE)
                .columns(
                Column("cameraId", "TEXT"),
                Column("worldId", "TEXT"),
                Column("ratio", "real"),
                Column("origin", "geometry"),
                Column("focalpoints", "geometry"),
                Column("fov", "INTEGER"),
                Column("skev_factor", "real"),
                Column("width", "integer"),
                Column("height", "integer")
            )
        )

        self.cur.execute(q1.get_sql())
        self.cur.execute(q2.get_sql())
        self.con.commit()

    def insert_cam(self, world_id: str, camera_node: Camera):
        cam = Table(CAMERA_TABLE)
        cam_id = camera_node.cam_id
        cam_ratio = camera_node.ratio
        lens = camera_node.lens
        focal_x = str(lens.focal_x)
        focal_y = str(lens.focal_y)
        cam_x, cam_y, cam_z = str(lens.cam_origin[0]), str(lens.cam_origin[1]), str(lens.cam_origin[2])
        camera_node.dimension = get_video_dimension(camera_node.video_file)
        width, height = camera_node.dimension

        q = (
            Query
                .into(cam)
                .insert(
                cam_id, world_id, cam_ratio,
                f"POINT Z ({cam_x} {cam_y} {cam_z})",
                f"POINT({focal_x} {focal_y})",
                lens.fov, lens.alpha,
                width, height
            )
        )
        # print(q)
        self.cur.execute(q.get_sql())
        self.con.commit()

    def retrieve_cam(self, query: Query = None, world_id: str = ""):
        """
        Called when executing update commands (add_camera, add_objs ...etc)
        """

        return query + self._select_cam_with_world_id(world_id) if query \
            else self._select_cam_with_world_id(world_id)  # UNION

    def _select_cam_with_world_id(self, world_id: str):
        """
        Select cams with certain world id
        """
        cam = Table(CAMERA_TABLE)
        q = (
            Query
                .from_(cam)
                .select("*")
                .where(cam.worldId == world_id)
        )
        # print(q)
        return q

    def filter_cam(self, query: Query, condition: str):
        """
        Called when executing filter commands (predicate, interval ...etc) 
        """
        return Query.from_(query).select("*").where(eval(condition))

    def get_cam(self, query: Query):
        """
        Execute sql command rapidly
        """

        # hack
        q = f"SELECT cameraId, ratio, ST_X(origin), ST_Y(origin), ST_Z(origin), ST_X(focalpoints), ST_Y(focalpoints), fov, skev_factor" + f" FROM ({query.get_sql()}) AS final"

        print(q)

        self.cur.execute(q)
        return self.cur.fetchall()


if __name__ == "__main__":
    # Let's define some attribute for constructing the world first
    name = "traffic_scene"  # world name
    units = "metrics"  # world units
    video_file = "../amber_videos/traffic-scene-shorter.mp4"  # example video file
    lens_attrs = {"fov": 120, "cam_origin": (0, 0, 0), "skew_factor": 0}
    point_attrs = {"p_id": "p1", "cam_id": "cam1", "x": 0, "y": 0, "z": 0, "time": None, "type": "pos"}
    camera_attrs = {"ratio": 0.5}
    fps = 30

    fov, res, cam_origin, skew_factor = (
        lens_attrs["fov"],
        [1280, 720],
        lens_attrs["cam_origin"],
        lens_attrs["skew_factor"],
    )

    cam_lens = lens.PinholeLens(res, cam_origin, fov, skew_factor)

    pt_id, cam_id, x, y, z, time, pt_type = (
        point_attrs["p_id"],
        point_attrs["cam_id"],
        point_attrs["x"],
        point_attrs["y"],
        point_attrs["z"],
        point_attrs["time"],
        point_attrs["type"],
    )
    location = point.Point(pt_id, cam_id, x, y, z, time, pt_type)

    ratio = camera_attrs["ratio"]

    # Ingest the camera to the world
    c1 = Camera(
        cam_id=cam_id,
        point=location,
        ratio=ratio,
        video_file=video_file,
        metadata_id=name + "_" + cam_id,
        lens=cam_lens,
    )

    db = Database()
    # db.insert_cam("1", 5, "1")
    # db.insert_cam("2", 3, "2")
    db.insert_cam("wid-1", c1)

    q = db.retrieve_cam(world_id="wid-1")
    res = db.execute_get_query(q)
    print(res)
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
