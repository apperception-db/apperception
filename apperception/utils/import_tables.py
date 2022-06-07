import pandas as pd
from psycopg2._psycopg import connection as Connection


def import_tables(conn: Connection, data_path: str):

    # Import CSV
    data_Cameras = pd.read_csv(data_path + "cameras.csv")
    df_Cameras = pd.DataFrame(data_Cameras)

    data_Item_General_Trajectory = pd.read_csv(data_path + "item_general_trajectory.csv")
    df_Item_General_Trajectory = pd.DataFrame(data_Item_General_Trajectory)

    data_General_Bbox = pd.read_csv(data_path + "general_bbox.csv")
    df_General_Bbox = pd.DataFrame(data_General_Bbox)

    # Connect to SQL Server
    cursor = conn.cursor()

    # Create Table
    cursor.execute("DROP TABLE IF EXISTS Cameras CASCADE;")
    cursor.execute("DROP TABLE IF EXISTS Item_General_Trajectory CASCADE;")
    cursor.execute("DROP TABLE IF EXISTS General_Bbox CASCADE;")

    cursor.execute(
        """
            CREATE TABLE Cameras (
                cameraId TEXT,
                frameId TEXT,
                frameNum Int,
                fileName TEXT,
                cameraTranslation geometry,
                cameraRotation real[4],
                cameraIntrinsic real[3][3],
                egoTranslation geometry,
                egoRotation real[4],
                timestamp timestamptz,
                cameraHeading real,
                egoHeading real
                )
    """
    )

    cursor.execute(
        """
            CREATE TABLE Item_General_Trajectory (
                itemId TEXT,
                cameraId TEXT,
                objectType TEXT,
                color TEXT,
                trajCentroids tgeompoint,
                largestBbox stbox,
                itemHeadings tfloat,
                PRIMARY KEY (itemId)
                )
    """
    )

    cursor.execute(
        """
            CREATE TABLE General_Bbox (
                itemId TEXT,
                cameraId TEXT,
                trajBbox stbox,
                FOREIGN KEY(itemId)
                    REFERENCES Item_General_Trajectory(itemId)
                )
    """
    )

    # Insert DataFrame to Table
    # for i,row in irisData.iterrows():
    #         sql = "INSERT INTO irisdb.iris VALUES (%s,%s,%s,%s,%s)"
    #         cursor.execute(sql, tuple(row))
    for i, row in df_Cameras.iterrows():
        cursor.execute(
            """
                    INSERT INTO Cameras (cameraId, frameId, frameNum, fileName, cameraTranslation, cameraRotation, cameraIntrinsic, egoTranslation, egoRotation, timestamp, cameraHeading, egoHeading)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    """,
            tuple(row),
        )
    cursor.execute(
        """
        CREATE INDEX ON Cameras (cameraId);
    """
    )

    for i, row in df_Item_General_Trajectory.iterrows():
        cursor.execute(
            """
                    INSERT INTO Item_General_Trajectory (itemId, cameraId, objectType, color, trajCentroids, largestBbox, itemHeadings)
                    VALUES (%s,%s,%s,%s,%s,%s,%s)
                    """,
            tuple(row),
        )
    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS traj_idx
        ON Item_General_Trajectory
        USING GiST(trajCentroids);
    """
    )
    cursor.execute(
        """
        CREATE INDEX ON Item_General_Trajectory (cameraId);
    """
    )

    for i, row in df_General_Bbox.iterrows():
        cursor.execute(
            """
                    INSERT INTO General_Bbox (itemId, cameraId, trajBbox)
                    VALUES (%s,%s,%s)
                    """,
            tuple(row),
        )
    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS item_idx
        ON General_Bbox(itemId);
    """
    )
    cursor.execute(
        """
        CREATE INDEX ON General_Bbox (cameraId);
    """
    )
    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS traj_bbox_idx
        ON General_Bbox
        USING GiST(trajBbox);
    """
    )

    conn.commit()