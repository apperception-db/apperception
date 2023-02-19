import os
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from apperception.database import Database


def import_tables(database: "Database", data_path: str):
    # Import CSV
    data_Cameras = pd.read_csv(os.path.join(data_path, "cameras.csv"))
    df_Cameras = pd.DataFrame(data_Cameras)

    data_Item_General_Trajectory = pd.read_csv(
        os.path.join(data_path, "item_general_trajectory.csv")
    )
    df_Item_General_Trajectory = pd.DataFrame(data_Item_General_Trajectory)

    data_General_Bbox = pd.read_csv(os.path.join(data_path, "general_bbox.csv"))
    df_General_Bbox = pd.DataFrame(data_General_Bbox)

    database.reset(False)

    for _, row in df_Item_General_Trajectory.iterrows():
        [print(type(e)) for e in row]

    # for _, row in df_Cameras.iterrows():
    #     database._insert_into_camera(row, False)

    # for _, row in df_Item_General_Trajectory.iterrows():
    #     database._insert_into_item_general_trajectory(row, False)

    # for _, row in df_General_Bbox.iterrows():
    #     database._insert_into_general_bbox(row, False)

    database._commit()
