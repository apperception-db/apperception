import datetime
import json
import os
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from pyquaternion import Quaternion

from apperception.data_types import Box

CREATE_ITEMTRAJ_SQL = """
CREATE TABLE IF NOT EXISTS Item_General_Trajectory(
    itemId TEXT,
    objectType TEXT,
    frameId TEXT,
    color TEXT,
    trajCentroids tgeompoint,
    largestBbox stbox,
    PRIMARY KEY (itemId)
);
"""

CREATE_BBOXES_SQL = """
CREATE TABLE IF NOT EXISTS General_Bbox(
    itemId TEXT,
    trajBbox stbox,
    FOREIGN KEY(itemId)
        REFERENCES Item_General_Trajectory(itemId)
);
"""

CREATE_CAMERA_SQL = """
CREATE TABLE IF NOT EXISTS Cameras(
    cameraId TEXT,
    worldId TEXT,
    frameId TEXT,
    frameNum Int,
    fileName TEXT,
    cameraTranslation geometry,
    cameraRotation real[4],
    cameraIntrinsic real[3][3],
    egoTranslation geometry,
    egoRotation real[4],
    timestamp TEXT
);
"""


def fetch_camera_config(scene_name, sample_data):
    """
    return
    [{
            camera_id: scene name,
            frame_id,
            frame_num: the frame sequence number
            filename: image file name,
            camera_translation,
            camera_rotation,
            camera_intrinsic(since it's a matrix, save as a nested array),
            ego_translation,
            ego_rotation,
            timestamp
    },
    ...
    ]
    """
    camera_config = []

    # TODO: different camera in one frame has same timestamp for same object
    # how to store same scene in different cameras
    all_frames = sample_data[
        (sample_data["scene_name"] == scene_name)
        # & (sample_data["filename"].str.contains("/CAM_FRONT/", regex=False))
    ]

    for idx, frame in all_frames.iterrows():
        config = {}
        config["camera_id"] = scene_name
        config["frame_id"] = frame["sample_token"]
        config["frame_num"] = frame["frame_order"]
        config["filename"] = frame["filename"]
        config["camera_translation"] = frame["camera_translation"]
        config["camera_rotation"] = frame["camera_rotation"]
        config["camera_intrinsic"] = frame["camera_intrinsic"]
        config["ego_translation"] = frame["ego_translation"]
        config["ego_rotation"] = frame["ego_rotation"]
        config["timestamp"] = frame["timestamp"]
        camera_config.append(config)

    return camera_config


# Create a camera table


def create_or_insert_camera_table(conn, world_name, camera):
    # Creating a cursor object using the cursor() method
    cursor = conn.cursor()
    """
    Create and Populate A camera table with the given camera object.
    """
    # Doping Cameras table if already exists.
    cursor.execute("DROP TABLE IF EXISTS Cameras")
    # Formal_Scenic_cameras table stands for the formal table which won't be erased
    # Test for now

    cursor.execute(CREATE_CAMERA_SQL)
    print("Camera Table created successfully........")
    insert_camera(
        conn,
        world_name,
        fetch_camera_config(camera.id, camera.object_recognition.sample_data),
    )
    return CREATE_CAMERA_SQL


# Helper function to insert the camera


def insert_camera(conn, world_name, camera_config):
    # Creating a cursor object using the cursor() method
    cursor = conn.cursor()
    values = []
    for config in camera_config:
        values.append(
            f"""(
                '{config['camera_id']}',
                '{world_name}',
                '{config['frame_id']}',
                {config['frame_num']},
                '{config['filename']}',
                'POINT Z ({' '.join(map(str, config['camera_translation']))})',
                ARRAY{config['camera_rotation']},
                ARRAY{config['camera_intrinsic']},
                'POINT Z ({' '.join(map(str, config['ego_translation']))})',
                ARRAY{config['ego_rotation']},
                '{config['timestamp']}'
            )"""
        )

    cursor.execute(
        f"""
        INSERT INTO Cameras (
            cameraId,
            worldId,
            frameId,
            frameNum,
            fileName,
            cameraTranslation,
            cameraRotation,
            cameraIntrinsic,
            egoTranslation,
            egoRotation,
            timestamp
        )
        VALUES {','.join(values)};
        """
    )

    print("New camera inserted successfully.........")
    conn.commit()


# create collections in db and set index for quick query


def insert_data(data_dir, db):
    with open(os.path.join(data_dir, "v1.0-mini", "sample_data.json")) as f:
        sample_data_json = json.load(f)
    db["sample_data"].insert_many(sample_data_json)
    db["sample_data"].create_index("token")
    db["sample_data"].create_index("filename")

    with open(os.path.join(data_dir, "v1.0-mini", "attribute.json")) as f:
        attribute_json = json.load(f)
    db["attribute"].insert_many(attribute_json)
    db["attribute"].create_index("token")

    with open(os.path.join(data_dir, "v1.0-mini", "calibrated_sensor.json")) as f:
        calibrated_sensor_json = json.load(f)
    db["calibrated_sensor"].insert_many(calibrated_sensor_json)
    db["calibrated_sensor"].create_index("token")

    with open(os.path.join(data_dir, "v1.0-mini", "category.json")) as f:
        category_json = json.load(f)
    db["category"].insert_many(category_json)
    db["category"].create_index("token")

    with open(os.path.join(data_dir, "v1.0-mini", "ego_pose.json")) as f:
        ego_pose_json = json.load(f)
    db["ego_pose"].insert_many(ego_pose_json)
    db["ego_pose"].create_index("token")

    with open(os.path.join(data_dir, "v1.0-mini", "instance.json")) as f:
        instance_json = json.load(f)
    db["instance"].insert_many(instance_json)
    db["instance"].create_index("token")

    with open(os.path.join(data_dir, "v1.0-mini", "sample_annotation.json")) as f:
        sample_annotation_json = json.load(f)
    db["sample_annotation"].insert_many(sample_annotation_json)
    db["sample_annotation"].create_index("token")

    with open(os.path.join(data_dir, "v1.0-mini", "frame_num.json")) as f:
        frame_num_json = json.load(f)
    db["frame_num"].insert_many(frame_num_json)
    db["frame_num"].create_index("token")


def transform_box(box: Box, camera):
    box.translate(-np.array(camera["egoTranslation"]))
    box.rotate(Quaternion(camera["egoRotation"]).inverse)

    box.translate(-np.array(camera["cameraTranslation"]))
    box.rotate(Quaternion(camera["cameraRotation"]).inverse)


# import matplotlib.pyplot as plt
# def overlay_bbox(image, corners):
# 	frame = cv2.imread(image)
# 	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
# 	for i in range(len(corners)):
# 		current_coner = (corners[0][i], corners[1][i])
# 		cv2.circle(frame,tuple([int(current_coner[0]), int(current_coner[1])]),4,(255,0,0),thickness=5)
# 	plt.rcParams["figure.figsize"] = (20,20)
# 	plt.figure()
# 	plt.imshow(frame)
# 	plt.show()


def recognize(scene_name, sample_data, annotation):
    """
    return:
    annotations: {
            object_id: {
                    bboxes: [[[x1, y1, z1], [x2, y2, z2]], ...]
                    object_type,
                    frame_num,
                    frame_id,
            }
            ...
    }
    """

    annotations = {}

    # TODO: different camera in one frame has same timestamp for same object
    # how to store same scene in different cameras
    img_files = sample_data[
        (sample_data["scene_name"] == scene_name)
        # & (sample_data["filename"].str.contains("/CAM_FRONT/", regex=False))
    ].sort_values(by="frame_order")

    for _, img_file in img_files.iterrows():
        # get bboxes and categories of all the objects appeared in the image file
        sample_token = img_file["sample_token"]
        frame_num = img_file["frame_order"]
        all_annotations = annotation[annotation["sample_token"] == sample_token]
        # camera_info = {}
        # camera_info['cameraTranslation'] = img_file['camera_translation']
        # camera_info['cameraRotation'] = img_file['camera_rotation']
        # camera_info['cameraIntrinsic'] = np.array(img_file['camera_intrinsic'])
        # camera_info['egoRotation'] = img_file['ego_rotation']
        # camera_info['egoTranslation'] = img_file['ego_translation']

        for _, ann in all_annotations.iterrows():
            item_id = ann["instance_token"]
            if item_id not in annotations:
                annotations[item_id] = {"bboxes": [], "frame_num": []}
                annotations[item_id]["object_type"] = ann["category"]

            box = Box(ann["translation"], ann["size"], Quaternion(ann["rotation"]))

            corners = box.corners()

            # if item_id == '6dd2cbf4c24b4caeb625035869bca7b5':
            # 	# print("corners", corners)
            # 	# transform_box(box, camera_info)
            # 	# print("transformed box: ", box.corners())
            # 	# corners_2d = box.map_2d(np.array(camera_info['cameraIntrinsic']))
            # 	corners_2d = transformation(box.center, camera_info)
            # 	print("2d_corner: ", corners_2d)
            # 	overlay_bbox("v1.0-mini/samples/CAM_FRONT/n015-2018-07-24-11-22-45+0800__CAM_FRONT__1532402927612460.jpg", corners_2d)

            bbox = [corners[:, 1], corners[:, 7]]
            annotations[item_id]["bboxes"].append(bbox)
            annotations[item_id]["frame_num"].append(int(frame_num))

    print("Recognization done, saving to database......")
    return annotations


def add_recognized_objs(conn, formatted_result, start_time, default_depth=True):
    clean_tables(conn)
    for item_id in formatted_result:
        object_type = formatted_result[item_id]["object_type"]
        recognized_bboxes = np.array(formatted_result[item_id]["bboxes"])
        tracked_cnt = formatted_result[item_id]["frame_num"]
        top_left = np.vstack(
            (recognized_bboxes[:, 0, 0], recognized_bboxes[:, 0, 1], recognized_bboxes[:, 0, 2])
        )
        # if default_depth:
        # 	top_left_depths = np.ones(len(recognized_bboxes))
        # else:
        # 	top_left_depths = self.__get_depths_of_points(recognized_bboxes[:,0,0], recognized_bboxes[:,0,1])

        # # Convert bottom right coordinates to world coordinates
        bottom_right = np.vstack(
            (recognized_bboxes[:, 1, 0], recognized_bboxes[:, 1, 1], recognized_bboxes[:, 1, 2])
        )
        # if default_depth:
        # 	bottom_right_depths = np.ones(len(tracked_cnt))
        # else:
        # 	bottom_right_depths = self.__get_depths_of_points(recognized_bboxes[:,1,0], recognized_bboxes[:,1,1])

        top_left = np.array(top_left.T)
        bottom_right = np.array(bottom_right.T)
        obj_traj = []
        for i in range(len(top_left)):
            current_tl = top_left[i]
            current_br = bottom_right[i]
            obj_traj.append([current_tl.tolist(), current_br.tolist()])

        bboxes_to_postgres(
            conn,
            item_id,
            object_type,
            "default_color",
            start_time,
            tracked_cnt,
            obj_traj,
            type="yolov4",
        )
        # bbox_to_tasm()


# Insert bboxes to postgres


def bboxes_to_postgres(
    conn, item_id, object_type, color, start_time, timestamps, bboxes, type="yolov3"
):
    if type == "yolov3":
        timestamps = range(timestamps)

    converted_bboxes = [bbox_to_data3d(bbox) for bbox in bboxes]
    pairs = []
    deltas = []
    for meta_box in converted_bboxes:
        pairs.append(meta_box[0])
        deltas.append(meta_box[1:])
    postgres_timestamps = convert_timestamps(start_time, timestamps)
    create_or_insert_general_trajectory(
        conn, item_id, object_type, color, postgres_timestamps, bboxes, pairs
    )
    # print(f"{item_id} saved successfully")


# Create general trajectory table
def create_or_insert_general_trajectory(
    conn, item_id, object_type, color, postgres_timestamps, bboxes, pairs
):
    cursor = conn.cursor()
    """
    Create and Populate A Trajectory table using mobilityDB.
    Now the timestamp matches, the starting time should be the meta data of the world
    Then the timestamp should be the timestamp regarding the world starting time
    """

    # Formal_Scenic_Item_General_Trajectory table stands for the formal table which won't be erased
    # Test for now

    cursor.execute(CREATE_ITEMTRAJ_SQL)
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS traj_idx ON Item_General_Trajectory USING GiST(trajCentroids);"
    )
    conn.commit()
    # Formal_Scenic_General_Bbox table stands for the formal table which won't be erased
    # Test for now

    cursor.execute(CREATE_BBOXES_SQL)
    cursor.execute("CREATE INDEX IF NOT EXISTS item_idx ON General_Bbox(itemId);")
    cursor.execute("CREATE INDEX IF NOT EXISTS traj_bbox_idx ON General_Bbox USING GiST(trajBbox);")
    conn.commit()
    # Insert the trajectory of the first item
    insert_general_trajectory(conn, item_id, object_type, color, postgres_timestamps, bboxes, pairs)


def join(array: Iterable, delim: str = ","):
    return delim.join(map(str, array))


# Insert general trajectory
def insert_general_trajectory(
    conn,
    item_id: str,
    object_type: str,
    color: str,
    postgres_timestamps: List[str],
    bboxes: List[
        List[List[float]]
    ],  # TODO: should be (float, float, float), (float, float, float))[]
    pairs: List[Tuple[float, float, float]],
):
    # Creating a cursor object using the cursor() method
    cursor = conn.cursor()

    # Inserting bboxes into Bbox table
    insert_bbox_trajectories_builder = []
    min_tl = np.full(3, np.inf)
    max_br = np.full(3, np.NINF)

    traj_centroids = []

    for timestamp, (tl, br), current_point in zip(postgres_timestamps, bboxes, pairs):
        min_tl = np.minimum(tl, min_tl)
        max_br = np.maximum(br, max_br)

        # Insert bbox
        insert_bbox_trajectories_builder.append(
            f"""
            INSERT INTO General_Bbox (itemId, trajBbox)
            VALUES (
                '{item_id}',
                STBOX 'STBOX ZT(
                    ({join([*tl, timestamp])}),
                    ({join([*br, timestamp])})
                )'
            );
            """
        )

        # Construct trajectory
        traj_centroids.append(f"POINT Z ({join(current_point, ' ')})@{timestamp}")

    # Insert the item_trajectory separately
    insert_trajectory = f"""
    INSERT INTO Item_General_Trajectory (itemId, objectType, color, trajCentroids, largestBbox)
    VALUES (
        '{item_id}',
        '{object_type}',
        '{color}',
        '{{{', '.join(traj_centroids)}}}',
        STBOX 'STBOX Z(
            ({join(min_tl)}),
            ({join(max_br)})
        )'
    );
    """

    cursor.execute(insert_trajectory)
    cursor.execute("".join(insert_bbox_trajectories_builder))

    # Commit your changes in the database
    conn.commit()


def transformation(copy_centroid_3d: np.ndarray, camera_config: Dict[str, Any]) -> np.ndarray:
    """
    TODO: transformation from 3d world coordinate to 2d frame coordinate given the camera config
    """
    centroid_3d: np.ndarray = np.copy(copy_centroid_3d)

    centroid_3d -= camera_config["egoTranslation"]
    centroid_3d = np.dot(
        Quaternion(camera_config["egoRotation"]).inverse.rotation_matrix, centroid_3d
    )

    centroid_3d -= camera_config["cameraTranslation"]
    centroid_3d = np.dot(
        Quaternion(camera_config["cameraRotation"]).inverse.rotation_matrix, centroid_3d
    )

    view = np.array(camera_config["cameraIntrinsic"])
    viewpad = np.eye(4)
    viewpad[: view.shape[0], : view.shape[1]] = view

    # Do operation in homogenous coordinates.
    centroid_3d = centroid_3d.reshape((3, 1))
    centroid_3d = np.concatenate((centroid_3d, np.ones((1, 1))))
    centroid_3d = np.dot(viewpad, centroid_3d)
    centroid_3d = centroid_3d[:3, :]

    centroid_3d = centroid_3d / centroid_3d[2:3, :].repeat(3, 0).reshape(3, 1)
    return centroid_3d[:2, :]


FetchCameraTuple = Tuple[
    str, List[float], List[float], List[float], List[float], List[List[float]], int, str
]


def fetch_camera(conn, scene_name, frame_timestamps) -> List["FetchCameraTuple"]:
    """
    TODO: Fix fetch camera that given a scene_name and frame_num, return the corresponding camera metadata
    scene_name: str
    frame_num: int[]
    return a list of metadata info for each frame_num
    """

    cursor = conn.cursor()
    # query = '''SELECT camera_info from camera_table where camera_table.camera_id == scene_name and camera_table.frame_num in frame_num'''
    # if cam_id == []:
    # 	query = '''SELECT cameraId, ratio, ST_X(origin), ST_Y(origin), ST_Z(origin), ST_X(focalpoints), ST_Y(focalpoints), fov, skev_factor ''' \
    # 	 + '''FROM Cameras WHERE worldId = \'%s\';''' %world_id
    # else:
    # 	query = '''SELECT cameraId, ratio, ST_X(origin), ST_Y(origin), ST_Z(origin), ST_X(focalpoints), ST_Y(focalpoints), fov, skev_factor ''' \
    # 	 + '''FROM Cameras WHERE cameraId IN (\'%s\') AND worldId = \'%s\';''' %(','.join(cam_id), world_id)
    # TODO: define ST_XYZ somewhere else
    query = f"""
    CREATE OR REPLACE FUNCTION ST_XYZ (g geometry) RETURNS real[] AS $$
        BEGIN
            RETURN ARRAY[ST_X(g), ST_Y(g), ST_Z(g)];
        END;
    $$ LANGUAGE plpgsql;

    SELECT
        cameraId,
        ST_XYZ(egoTranslation),
        egoRotation,
        ST_XYZ(cameraTranslation),
        cameraRotation,
        cameraIntrinsic,
        frameNum,
        fileName,
        cameraHeading,
        egoHeading
    FROM Cameras
    WHERE
        cameraId = '{scene_name}' AND
        timestamp IN ({",".join(map(str, frame_timestamps))})
    ORDER BY cameraId ASC, frameNum ASC;
    """
    # print(query)
    cursor.execute(query)
    return cursor.fetchall()

def fetch_camera_framenum(conn, scene_name, frame_nums) -> List["FetchCameraTuple"]:
    """
    TODO: Fix fetch camera that given a scene_name and frame_num, return the corresponding camera metadata
    scene_name: str
    frame_num: int[]
    return a list of metadata info for each frame_num
    """

    cursor = conn.cursor()
    # query = '''SELECT camera_info from camera_table where camera_table.camera_id == scene_name and camera_table.frame_num in frame_num'''
    # if cam_id == []:
    # 	query = '''SELECT cameraId, ratio, ST_X(origin), ST_Y(origin), ST_Z(origin), ST_X(focalpoints), ST_Y(focalpoints), fov, skev_factor ''' \
    # 	 + '''FROM Cameras WHERE worldId = \'%s\';''' %world_id
    # else:
    # 	query = '''SELECT cameraId, ratio, ST_X(origin), ST_Y(origin), ST_Z(origin), ST_X(focalpoints), ST_Y(focalpoints), fov, skev_factor ''' \
    # 	 + '''FROM Cameras WHERE cameraId IN (\'%s\') AND worldId = \'%s\';''' %(','.join(cam_id), world_id)
    # TODO: define ST_XYZ somewhere else
    query = f"""
    CREATE OR REPLACE FUNCTION ST_XYZ (g geometry) RETURNS real[] AS $$
        BEGIN
            RETURN ARRAY[ST_X(g), ST_Y(g), ST_Z(g)];
        END;
    $$ LANGUAGE plpgsql;

    SELECT
        cameraId,
        ST_XYZ(egoTranslation),
        egoRotation,
        ST_XYZ(cameraTranslation),
        cameraRotation,
        cameraIntrinsic,
        frameNum,
        fileName,
        cameraHeading,
        egoHeading
    FROM Cameras
    WHERE
        cameraId = '{scene_name}' AND
        frameNum IN ({",".join(map(str, frame_nums))})
    ORDER BY cameraId ASC, frameNum ASC;
    """
    # print(query)
    cursor.execute(query)
    return cursor.fetchall()

def timestamp_to_framenum(conn, scene_name: str, timestamps: List[str]):
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

def clean_tables(conn):
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS General_Bbox;")
    cursor.execute("DROP TABLE IF EXISTS Item_General_Trajectory;")
    conn.commit()


def export_tables(conn):
    # create a query to specify which values we want from the database.
    s = "SELECT *"
    s += " FROM "
    s_trajectory = s + "Item_General_Trajectory"
    s_bbox = s + "General_Bbox"
    s_camera = s + "Cameras"

    # set up our database connection.
    db_cursor = conn.cursor()

    # Use the COPY function on the SQL we created above.
    SQL_trajectory_output = "COPY ({0}) TO STDOUT WITH CSV HEADER".format(s_trajectory)
    SQL_bbox_output = "COPY ({0}) TO STDOUT WITH CSV HEADER".format(s_bbox)
    SQL_camera_output = "COPY ({0}) TO STDOUT WITH CSV HEADER".format(s_camera)

    # Set up a variable to store our file path and name.
    trajectory_file = "test_trajectory.csv"
    with open(trajectory_file, "w") as trajectory_output:
        db_cursor.copy_expert(SQL_trajectory_output, trajectory_output)

    bbox_file = "test_bbox.csv"
    with open(bbox_file, "w") as bbox_output:
        db_cursor.copy_expert(SQL_bbox_output, bbox_output)

    camera_file = "test_camera.csv"
    with open(camera_file, "w") as camera_output:
        db_cursor.copy_expert(SQL_camera_output, camera_output)


def import_tables(conn, data_path):
    # # Old Version:
    # cur = conn.cursor()
    # cur.execute(CREATE_CAMERA_SQL)
    # cur.execute(CREATE_ITEMTRAJ_SQL)
    # cur.execute(CREATE_BBOXES_SQL)
    # conn.commit()
    # with open("test_camera.csv", "r") as camera_f:
    #     cur.copy_expert(file=camera_f, sql="COPY Cameras FROM STDIN CSV HEADER DELIMITER as ','")
    # with open("test_trajectory.csv", "r") as trajectory_f:
    #     cur.copy_expert(
    #         file=trajectory_f,
    #         sql="COPY Item_General_Trajectory FROM STDIN CSV HEADER DELIMITER as ','",
    #     )
    # with open("test_bbox.csv", "r") as bbox_f:
    #     cur.copy_expert(file=bbox_f, sql="COPY General_Bbox FROM STDIN CSV HEADER DELIMITER as ','")

    # conn.commit()

    # Current Version:
    # Import CSV
    data_Cameras = pd.read_csv(r"test_camera.csv")
    df_Cameras = pd.DataFrame(data_Cameras)

    data_Item_General_Trajectory = pd.read_csv(r"test_trajectory.csv")
    df_Item_General_Trajectory = pd.DataFrame(data_Item_General_Trajectory)

    data_General_Bbox = pd.read_csv(r"test_bbox.csv")
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

    for i, row in df_Item_General_Trajectory.iterrows():
        cursor.execute(
            """
                    INSERT INTO Item_General_Trajectory (itemId, cameraId, objectType, color, trajCentroids, largestBbox, itemHeadings)
                    VALUES (%s,%s,%s,%s,%s,%s,%s)
                    """,
            tuple(row),
        )

    for i, row in df_General_Bbox.iterrows():
        cursor.execute(
            """
                    INSERT INTO General_Bbox (itemId, cameraId, trajBbox)
                    VALUES (%s,%s,%s)
                    """,
            tuple(row),
        )

    conn.commit()


# Helper function to convert the timestam to the timestamp formula pg-trajectory uses


def convert_timestamps(start_time: datetime.datetime, timestamps: Iterable[int]):
    return [str(start_time + datetime.timedelta(seconds=t)) for t in timestamps]


# Helper function to convert trajectory to centroids


def bbox_to_data3d(bbox: List[List[float]]):
    """
    Compute the center, x, y, z delta of the bbox
    """
    tl, br = bbox
    x_delta = (br[0] - tl[0]) / 2
    y_delta = (br[1] - tl[1]) / 2
    z_delta = (br[2] - tl[2]) / 2
    center = (tl[0] + x_delta, tl[1] + y_delta, tl[2] + z_delta)

    return center, x_delta, y_delta, z_delta
