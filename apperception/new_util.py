import lens
import point
from video_context import Camera


def create_camera(cam_id, fov):
    # Let's define some attribute for constructing the world first
    name = "traffic_scene"  # world name
    video_file = "./amber_videos/traffic-scene-mini.mp4"  # example video file
    lens_attrs = {"fov": fov, "cam_origin": (0, 0, 0), "skew_factor": 0}
    point_attrs = {
        "p_id": "p1",
        "cam_id": cam_id,
        "x": 0,
        "y": 0,
        "z": 0,
        "time": None,
        "type": "pos",
    }
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
    return Camera(
        cam_id=cam_id,
        point=location,
        ratio=ratio,
        video_file=video_file,
        metadata_id=name + "_" + cam_id,
        lens=cam_lens,
    )
