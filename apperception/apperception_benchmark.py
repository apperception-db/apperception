# IMPORTS

from bounding_box import BoundingBox
import lens
import point
from world import World, world_executor

# import tasm

# Let's define some attribute for constructing the world first
name = "traffic_scene"  # world name
units = "metrics"  # world units
video_file = "./amber_videos/traffic-scene-shorter.mp4"  # example video file
lens_attrs = {"fov": 120, "cam_origin": (0, 0, 0), "skew_factor": 0}
point_attrs = {"p_id": "p1", "cam_id": "cam1", "x": 0, "y": 0, "z": 0, "time": None, "type": "pos"}
camera_attrs = {"ratio": 0.5}
fps = 30

# First we define a world
traffic_world = World(name=name, units=units)
world_executor.connect_db(user="docker", password="docker", database_name="mobilitydb")
conn = world_executor.conn
cursor = conn.cursor()
cursor.execute("DROP TABLE IF EXISTS Worlds;")
cursor.execute("DROP TABLE IF EXISTS Cameras;")
conn.commit()

# Secondly we construct the camera
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
traffic_world = traffic_world.camera(
    cam_id=cam_id,
    location=location,
    ratio=ratio,
    video_file=video_file,
    metadata_identifier=name + "_" + cam_id,
    lens=cam_lens,
)

# Call execute on the world to run the detection algorithm and save the real data to the database
recognized_world = traffic_world.recognize(cam_id, recognition_area=BoundingBox(0, 50, 50, 100))
recognized_world = recognized_world.recognize(cam_id, recognition_area=BoundingBox(0, 0, 100, 50))
recognized_world.execute()

volume = traffic_world.select_intersection_of_interest_or_use_default(cam_id=cam_id)
filtered_world = traffic_world.predicate(lambda obj: obj.object_type == "car").predicate(
    lambda obj: obj.location in volume, {"volume": volume}
)
filtered_world = filtered_world.interval([0, fps * 3])

# to get the trajectory and the video over the entire trajectory(amber case)
filtered_ids = filtered_world.selectkey(distinct=True).execute()
print("filtered_ids are", filtered_ids)
print(len(filtered_ids))
if len(filtered_ids) > 0:
    id_array = [e[0] for e in filtered_ids]
    # Fetch the trajectory of these items
    trajectory = (
        traffic_world.predicate(lambda obj: obj.object_id in id_array, {"id_array": id_array})
        .get_trajectory(distinct=True)
        .execute()
    )
    traffic_world.overlay_trajectory(cam_id, trajectory)
    # Get the videos of these items
#     entire_video = traffic_world.predicate(lambda obj: obj.object_id in id_array, {"id_array":id_array}).get_video()
#     entire_video.execute()
