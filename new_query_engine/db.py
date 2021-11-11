import sqlite3
import psycopg2
from pypika import Query, Table, Field

class Database:
    def __init__(self):
        # should setup a postgres in docker first
        self.con = psycopg2.connect(dbname = "postgres", user = "postgres", host = "localhost", port = "5432", password = "1234")
        self.cur = self.con.cursor()

        # create camera table
        self.cur.execute("DROP TABLE IF EXISTS cam")
        self.cur.execute("CREATE TABLE cam (cam_id text, size int, world_id text)")
        self.con.commit()

    def insert_cam(self, cam_id: str, cam_size: int, world_id: str):
        cam = Table("cam")
        q = Query.into(cam).insert(cam_id, cam_size, world_id)
        self.cur.execute(q.get_sql())
        self.con.commit()

    def concat_with(self, query: Query, world_id: str):
        """
        Called when executing update commands (add_camera, add_objs ...etc)
        """
        return query + self._select_cam_with_world_id(world_id) if query \
                else self._select_cam_with_world_id(world_id) # UNION

    def nest_from(self, sub_query: Query, condition: str):
        """
        Called when executing filter commands (predicate, interval ...etc) 
        """
        return Query.from_(sub_query).select("*").where(eval(condition))

    def execute_get_query(self, query: Query):
        """
        Execute sql command rapidly
        """
        self.cur.execute(query.get_sql())
        return self.cur.fetchall()

    def _select_cam_with_world_id(self, world_id: str):
        """
        Select cams with certain world id
        """
        cam = Table("cam")
        q = Query.from_(cam).select("*").where(cam.world_id == world_id)
        return q

if __name__ == "__main__":
    db = Database()
    db.insert_cam("1", 5, "1")
    db.insert_cam("2", 3, "2")
    
    # w1 = world()
    q = ""

    # w2 = w1.add_camera({cam_id: "1", size: 5, world_id: "1"})
    q = db.concat_with(query = q, world_id = "1")
    print(q)

    # w3 = w2.predicate(cam.size < 4)
    q = db.nest_from(sub_query = q, condition = "sub_query.size < 4")
    print(q)

    # w4 = w3.add_camera({cam_id: "1", size: 5, world_id: "2"})
    q = db.concat_with(query = q, world_id = "2")
    print(q)

    # w5 = w4.predicate(cam.size < 6)
    q = db.nest_from(sub_query = q, condition = "sub_query.size < 6")
    print(q)

    # w6 = w5.get_camera()
    res = db.execute_get_query(query = q)

    print(list(res))