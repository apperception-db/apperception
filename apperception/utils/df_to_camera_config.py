from apperception.data_types import CameraConfig


def df_to_camera_config(scene_name: str, sample_data):
    all_frames = sample_data[(sample_data["scene_name"] == scene_name)].sort_values(
        by="frame_order"
    )

    return [
        CameraConfig(
            frame_id=frame.token,
            frame_num=int(frame.frame_order),
            filename=frame.filename,
            camera_translation=frame.camera_translation,
            camera_rotation=frame.camera_rotation,
            camera_intrinsic=frame.camera_intrinsic,
            ego_translation=frame.ego_translation,
            ego_rotation=frame.ego_rotation,
            timestamp=frame.timestamp,
            cameraHeading=frame.camera_heading,
            egoHeading=frame.ego_heading,
        )
        for frame in all_frames.itertuples(index=False)
    ]
