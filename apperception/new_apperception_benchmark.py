import lens
import point
from bounding_box import BoundingBox
from new_world import World


# Let's define some attribute for constructing the world first
name = "traffic_scene"  # world name
units = "metrics"  # world units
video_file = "./amber_videos/traffic-scene-shorter.mp4"  # example video file
lens_attrs = {"fov": 120, "cam_origin": (0, 0, 0), "skew_factor": 0}
point_attrs = {"p_id": "p1", "cam_id": "cam1", "x": 0, "y": 0, "z": 0, "time": None, "type": "pos"}
camera_attrs = {"ratio": 0.5}
fps = 30

# 1. define a world
traffic_world = World()  # TODO: name, units?

# 2. construct a camera
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

# ingest the camera into the world
traffic_world = traffic_world.add_camera(
    cam_id=cam_id,
    location=location,
    ratio=ratio,
    video_file=video_file,
    metadata_identifier=name + "_" + cam_id,
    lens=cam_lens,
)

# Call execute on the world to run the detection algorithm and save the real data to the database
recognized_world = traffic_world.recognize(cam_id)

# TODO: filtered by volume
volume = traffic_world.select_intersection_of_interest_or_use_default(cam_id=cam_id)
# filtered_world = traffic_world.predicate(lambda obj: obj.object_type == "car").predicate(
#     lambda obj: obj.location in volume, {"volume": volume}
# )
# filtered_world = filtered_world.interval([0, fps * 3])

# filtered_world = recognized_world.filter_traj_type("car").filter_traj_volume(volume).interval(0, fps * 3)
filtered_world = recognized_world.filter_traj_type("car").interval(0, fps * 3)
filtered_ids = filtered_world.get_traj_key()

print("filtered_ids are", filtered_ids)
print(len(filtered_ids))

# trajectory = recognized_world.filter_traj_type("car").filter_traj_volume(volume).get_traj()
trajectory = recognized_world.filter_traj_type("car").get_traj()
print("trajectories are", trajectory)
traffic_world.overlay_trajectory(cam_id, trajectory)
