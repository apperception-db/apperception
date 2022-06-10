import psycopg2
import json

def get_lanesection_id_from_point(cur, x, y):
    query = """
        select lanesection.id from lanesection
        where lanesection.id in
        (select elementId from polygon where st_contains(elementpolygon, ST_Point({}, {})));
        """.format(x, y)

    cur.execute(query)
    return cur.fetchall()

def is_right_lanesection(cur, lanesecid1, lanesecid2):
    query = """
        select count(*)
        from lanesection as l
        where l.id = '{}' and l.laneToRight = '{}'
    """.format(lanesecid1, lanesecid2)
    cur.execute(query)
    return cur.fetchall()[0][0] != 0

def is_left_lanesection(cur, lanesecid1, lanesecid2):
    query = """
        select count(*)
        from lanesection as l
        where l.id = '{}' and l.laneToLeft= '{}'
    """.format(lanesecid1, lanesecid2)
    cur.execute(query)
    return cur.fetchall()[0][0] != 0

def main():
    con = psycopg2.connect(
        dbname="mobilitydb", user="docker", host="localhost", port="25432", password="docker"
    )
    cur = con.cursor()
    print(get_lanesection_id_from_point(cur, 1321.67, 1461.85205466148))
    print(is_right_lanesection(cur, "0f98559f-b844-424a-bfc5-8f8b19aa3724_sec", "4e13a1c5-4769-4e64-a03b-affaf90f7289_sec"))
    # cur.execute("select asMFJSON(trajcentroids)::json from item_general_trajectory where objecttype = 'vehicle.car'")
    # records = cur.fetchall()

    # centroid_strs = []

    # for record in records:
    #     d = record[0]
    #     seq_dict = d["sequences"][0]
    #     coordinates = seq_dict["coordinates"]
    #     timestamps = seq_dict["datetimes"]

    #     tgeompoint = []

    #     for i in range(len(coordinates)):
    #         # (x, y, z) -> (x, y)
    #         coordinates[i] = coordinates[i][:2]

    #         tgeompoint.append("Point({} {})@{}".format(
    #             coordinates[i][0], coordinates[i][1], timestamps[i]))

    #     import pdb;pdb.set_trace()
    #     centroid_strs.append("'[{}]'".format(", ".join(tgeompoint)))


    # for cnetroid_str in centroid_strs:
    #     cur.execute()


if __name__ == "__main__":
    main()
