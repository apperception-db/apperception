import os
import pickle

from apperception.database import Database
from apperception.data_types import Camera, CameraConfig


def import_pickle(database: "Database", data_path: str):
    with open(os.path.join(data_path, "frames.pkl"), "rb") as f:
        data_frames = pickle.loads(f.read())

    database.reset(True)
    for scene, val in data_frames.items():
        scene_info = scene.split("-")
        scene_id = scene_info[0] + "-" + scene_info[1]
        configs = [
            CameraConfig(
                frame_id=frame[1],
                frame_num=int(frame[2]),
                filename=frame[3],
                camera_translation=frame[4],
                camera_rotation=frame[5],
                camera_intrinsic=frame[6],
                ego_translation=frame[7],
                ego_rotation=frame[8],
                timestamp=frame[9],
                cameraHeading=frame[10],
                egoHeading=frame[11],
            )
            for frame in val["frames"]
        ]
        camera = Camera(config=configs, id=scene_id)
        database.insert_cam(camera)

    database._commit()
