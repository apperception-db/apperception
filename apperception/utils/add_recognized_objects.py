from datetime import datetime
from typing import TYPE_CHECKING, Dict, List, Tuple

import numpy as np

from .bbox_to_data3d import bbox_to_data3d
from .join import join

if TYPE_CHECKING:
    from psycopg2 import connection as Connection

    from ..data_types import TrackedObject


def add_recognized_objects(
    conn: "Connection", formatted_result: Dict[str, "TrackedObject"], camera_id: str
):
    for item_id in formatted_result:
        object_type = formatted_result[item_id].object_type
        recognized_bboxes = np.array(formatted_result[item_id].bboxes)
        timestamps = formatted_result[item_id].timestamps
        itemHeading_list = formatted_result[item_id].itemHeading

        top_left = recognized_bboxes[:, 0, :]
        bottom_right = recognized_bboxes[:, 1, :]

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
            timestamps,
            obj_traj,
            camera_id,
            itemHeading_list,
        )


def bboxes_to_postgres(
    conn: "Connection",
    item_id: str,
    object_type: str,
    color: str,
    timestamps: List[int],
    bboxes: List[List[List[float]]],
    camera_id: str,
    itemHeading_list: List[int],
):
    converted_bboxes = [bbox_to_data3d(bbox) for bbox in bboxes]
    pairs = []
    deltas = []
    for meta_box in converted_bboxes:
        pairs.append(meta_box[0])
        deltas.append(meta_box[1:])
    insert_general_trajectory(
        conn,
        item_id,
        object_type,
        color,
        [str(datetime.fromtimestamp(t / 1000000.0)) for t in timestamps],
        bboxes,
        pairs,
        camera_id,
        itemHeading_list,
    )
    # print(f"{item_id} saved successfully")


# Insert general trajectory
def insert_general_trajectory(
    conn,
    item_id: str,
    object_type: str,
    color: str,
    postgres_timestamps: List[str],
    bboxes: List[
        List[List[float]]
    ],  # TODO: should be ((float, float, float), (float, float, float))[]
    pairs: List[Tuple[float, float, float]],
    camera_id: str,
    itemHeading_list: List[int],
):

    # Creating a cursor object using the cursor() method
    cursor = conn.cursor()

    # Inserting bboxes into Bbox table
    insert_bbox_trajectories_builder = []
    min_tl = np.full(3, np.inf)
    max_br = np.full(3, np.NINF)

    traj_centroids = []
    itemHeadings = []
    prevTimestamp = None
    for timestamp, (tl, br), current_point, curItemHeading in zip(
        postgres_timestamps, bboxes, pairs, itemHeading_list
    ):
        if prevTimestamp == timestamp:
            continue
        prevTimestamp = timestamp
        min_tl = np.minimum(tl, min_tl)
        max_br = np.maximum(br, max_br)

        # Insert bbox
        insert_bbox_trajectories_builder.append(
            f"""(
            '{item_id}',
            '{camera_id}',
            STBOX 'STBOX ZT(
                ({join([*tl, timestamp])}),
                ({join([*br, timestamp])})
            )',
            timestamptz '{timestamp}'
        )"""
        )

        # Construct trajectory
        traj_centroids.append(f"POINT Z ({join(current_point, ' ')})@{timestamp}")
        itemHeadings.append(f"{curItemHeading}@{timestamp}")

    # Insert the item_trajectory separately
    insert_trajectory = f"""
    INSERT INTO Item_General_Trajectory (itemId, cameraId, objectType, color, trajCentroids, largestBbox, itemHeadings)
    VALUES (
        '{item_id}',
        '{camera_id}',
        '{object_type}',
        '{color}',
        tgeompoint '{{[{', '.join(traj_centroids)}]}}',
        STBOX 'STBOX Z(
            ({join(min_tl)}),
            ({join(max_br)})
        )',
        tfloat '{{[{', '.join(itemHeadings)}]}}'
    );
    """

    cursor.execute(insert_trajectory)
    if len(insert_bbox_trajectories_builder):
        cursor.execute(
            f"INSERT INTO General_Bbox (itemId, cameraId, trajBbox, timestamp) VALUES {','.join(insert_bbox_trajectories_builder)}"
        )

    # Commit your changes in the database
    conn.commit()
