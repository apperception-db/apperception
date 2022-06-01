from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from psycopg2 import connection as Connection


def timestamp_to_framenum(conn: "Connection", scene_name: str, timestamps: List[str]):
    cursor = conn.cursor()
    query = f"""
    SELECT
        DISTINCT frameNum
    FROM Cameras
    WHERE
        cameraId = '{scene_name}' AND
        timestamp IN ({",".join(map(str, timestamps))})
    ORDER BY frameNum ASC;
    """
    # print(query)
    cursor.execute(query)
    return cursor.fetchall()
