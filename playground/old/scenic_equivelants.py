# %%
# Deleting the .apperception_cache if it exists, as to avoid DB conflict errors
import os
import shutil

dirpath = os.path.join('.apperception_cache')
if os.path.exists(dirpath) and os.path.isdir(dirpath):
    shutil.rmtree(dirpath)

dirpath = os.path.join('output')
if os.path.exists(dirpath) and os.path.isdir(dirpath):
    shutil.rmtree(dirpath)
os.mkdir(dirpath)

# This piece of code is unsafe, and should not be run if not needed. 
# It serves for test purposes when one recieves a "dead kernel" error.
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# %%
import sys
sys.path.append(os.path.join(os.getcwd(),"apperception"))

### IMPORTS
import lens
import point
from new_world import empty_world

# Let's define some attribute for constructing the world first
name = "trafficScene"  # world name
units = "metrics"  # world units
video_file = "amber_videos/traffic-scene.mp4"  # example video file
lens_attrs = {"fov": 120, "cam_origin": (0, 0, 0), "skew_factor": 0}
point_attrs = {"p_id": "p1", "cam_id": "cam1", "x": 0, "y": 0, "z": 0, "time": None, "type": "pos"}
camera_attrs = {"ratio": 0.5}
fps = 30

# 1. define a world
traffic_world = empty_world(name)

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
location = point.Point(pt_id, cam_id, (x, y, z), time, pt_type)

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

volume = traffic_world.select_intersection_of_interest_or_use_default(cam_id=cam_id)

# %%
cams = traffic_world.get_camera()
lens = traffic_world.get_len()
# ids = traffic_world.get_id()
print("cameras are", cams)
print("lens are", lens)
# print("ids are", ids)

# %%
import time
start_time = time.time()
### Scenic Code ###
# ego = Car
# Car offset by (Range(-10, 10), Range(20, 40))

### Apperception Query ###
filtered_world = recognized_world.filter_traj_type("car")

## OPTION 1 ###
# filtered_world = filtered_world.filter_relative_to_type(x_range=(-10, 10), y_range=(-1, 5), z_range=(-10, 0),
#                                                         type="camera")
# The idea is that the user passes in a lambda function, that specifies the relationship that must be met between the queried
# object, and some object of the type passed to the function. In this case, the lambda function filters such that the offset 
# is between -10 and 10 in the x direction, and between 20 and 40 in the y direction, relative to some camera.

### OPTION 2 ###
filtered_world = filtered_world.filter_pred_relative_to_type(pred = lambda obj: (cam.x - 10) <= obj.x <= (cam.x + 10) and (cam.y - 15) <= obj.y <= (cam.y + 70))
# The idea is that filter_offset_type() takes in two arguments: the offset in terms of coordinates, a relative heading 
# as well as the type of object to be offset from. In this case, we want it to be somehwere between -10 and 10 units
# offset relative to a camera's x position, somehwere between 20 and 40 units offset relative to some camera's y position, 
# and we dont care about the offset relative to the camera's z position. We also dont care about the relative heading difference.

filtered_ids = filtered_world.get_traj_key()
print("filtered_ids are", filtered_ids)

print("----------------------------------------------------------------------")
print("Total execution time is: %s seconds" % (time.time() - start_time))
print("Device Details: \n Processor: AMD Ryzen 7 5800H \n RAM Size: 16GB \n Graphics Card: NVIDIA GeForce RTX 3060 Laptop")
print("----------------------------------------------------------------------")
# traffic-scene-shorter (length of 4 seconds): runtime of 81.82859063148499 seconds
# traffic-scene (length of 20 seconds): runtime of 98.58345794677734 seconds


# %%
filtered_world.get_video([cam_id], boxed=True)

# %%
# ### Scenic Code ###
# # ego = Car
# # Car

# ### Apperception Query ###
# filtered_world = recognized_world.filter_traj_type("car").interval(0, fps * 3)
# filtered_ids = filtered_world.get_traj_key()
# print("filtered_ids are", filtered_ids)

# # render tracking video
# filtered_world.get_video([cam_id])

# %%
# ### Scenic Code ###
# # ego = Car
# # Car offset by (Range(-10, 10), Range(20, 40)), 
# # 	facing Range(-5, 5) deg

# ### Apperception Query ###
# filtered_world = recognized_world.filter_traj_type("car")

# ## OPTION 1 ###
# filtered_world = filtered_world.filter_relative_to_type(relative=lambda obj, camera: -10 <= (camera.x - obj.x) <= 10 \
#                                                                                   and 20 <= (camera.y - obj.y) <= 40,
#                                                         type="camera")

# ### OPTION 2 ###
# filtered_world = filtered_world.filter_relative_to_type(offset=((-10, 10), (20, 40), None), heading=None, type="camera")

# filtered_world = filtered_world.filter_heading(-5, 5)
# # Filters for objects that have heading between -5 and 5 degrees

# filtered_ids = filtered_world.get_traj_key()
# print("filtered_ids are", filtered_ids)

# %%
# ### Scenic Code ###
# # ego = Car
# # Car offset by (Range(-10, 10), Range(20, 40)), 
# # 	facing Range(-5, 5) deg relative to ego

# ###  Query ###
# filtered_world = recognized_world.filter_traj_type("car")

# ## OPTION 1 ###
# filtered_world = filtered_world.filter_relative_to_type(relative=lambda obj, camera: -10 <= (camera.x - obj.x) <= 10 \
#                                                                                   and 20 <= (camera.y - obj.y) <= 40 \
#                                                                                   and -5 <= (camera.heading - obj.heading) <= 5,
#                                                         type="camera")
# # Now filtering for a relative heading between -5 and 5 degrees

# ### OPTION 2 ###
# filtered_world = filtered_world.filter_relative_to_type(offset=((-10, 10), (20, 40), None), heading=(-5, 5), type="camera")
# # Now filtering for a relative heading between -5 and 5 degrees

# filtered_ids = filtered_world.get_traj_key()
# print("filtered_ids are", filtered_ids)

# %%
# ### Scenic Code ###
# # ego = Car
# # Car left of ego by 0.25 

# ### Apperception Query ###
# filtered_world = recognized_world.filter_traj_type("car")

# ## OPTION 1 ###
# def left_of(obj, camera):
#     expec_x = obj.x + 0.25 * np.cos(camera.heading)
#     expec_y = obj.y - 0.25 * np.sin(camera.heading)
#     # Should also allow some sort of variation, to account for noise (and since exact equality is unlikley)
#     return (expec_x == camera.x) and (expec_y == camera.y)

# filtered_world = filtered_world.filter_relative_to_type(relative=left_of, type="camera")
# # Now filtering such that the car is left of ego by 0.25 units

# ### OPTION 2 ##
# # Not possible

# filtered_ids = filtered_world.get_traj_key()
# print("filtered_ids are", filtered_ids)

# %%
# ### Scenic Code ###
# # ego = Car
# # badAngle = Range(10, 20) deg
# # Car left of ego by 0.25,
# # 	facing badAngle relative to ego

# ### Apperception Query ###
# filtered_world = recognized_world.filter_traj_type("car")

# ## OPTION 1 ###
# def filter(obj, camera):
#     expec_x = obj.x + 0.25 * np.cos(camera.heading)
#     expec_y = obj.y - 0.25 * np.sin(camera.heading)
#     # Should also allow some sort of variation, to account for noise (and since exact equality is unlikley)
#     return (expec_x == camera.x) and (expec_y == camera.y) and 10 <= (camera.heading - obj.heading) <= 20

# filtered_world = filtered_world.filter_relative_to_type(relative=filter, type="camera")
# # Now filtering such that the car is left of ego by 0.25 units

# ### OPTION 2 ##
# # Not possible

# filtered_ids = filtered_world.get_traj_key()
# print("filtered_ids are", filtered_ids)

# %%
# def roadDirection(x, y, z):
#     # TODO: Implement
#     # Returns the direction (in 360 degree angle form) of the road at the coordinates (x, y, z)
#     # If their is no such road, returns a value of None
#     return None

# %%
#### FURTHER QUERIES WILL USE THE OPTION 1 LISTED ABOVE ####

# %%
### Scenic Code ###
# weather = Uniform("sunny", "rainy", "thunder")
# time = Range(10, 12)
#
# ego = car on road
# otherCar = Car ahead of ego by Range(4, 19)
# require not (otherCar in intersection)

# %%
### Scenic Code ###
# spot = OrientedPoint on curb
# ego = Car at (spot offset by (Range(2,4), Range(5,10)))
# sideCar = Car left of spot by Range(1,3)

# %%
### Scenic Code ###
# def placeObjs(car, numCars):
#     for i in range(numCars):
#         car = Car ahead of car by Range(4, 5)
#         leftCar = Car left of car by Normal(2, 0.1), facing roadDirection
#         rightCar = Car right of car by Normal(3, 0.1), facing Range(0, 10) deg relative to ego.heading
#     return leftCar, rightCar

# spawn_point = 207.26 @ 8.72
# ego = Car at spawn_point, with visible_distance 200

# leftCar, rightCar = placeObjs(ego, 2)
# require (distance to leftCar) < 200
# require (distance to rightCar) < 200


# %%
### Scenic Code ###
# def placeObjs(numPeds):
#     for i in range(numPeds):
#         Pedestrian offset by Range(-5, 5) @ Range(0, 200),
#             facing Range(-120, 120) deg relative to ego.heading

# spawn_point = 207.26 @ 8.72
# ego = Car at spawn_point,
#         with visibleDistance 200

# placeObjs(3)

# %%
### Scenic Code ###
# ego = Car on drivableRoad,
#         facing Range(-15, 15) deg relative to roadDirection,
#         with visibleDistance 50, 
#         with viewAngle 135 deg
# ped = Pedestrian on roadsOrIntersections,
#         with regionContainedIn roadRegion,
#         facing Range(-180, 180) deg

# require abs(relative heading of ped from ego) > 70 deg

# %%
### Scenic Code ###
# offset = Uniform(-1, 1) * Range(90, 180) deg

# ego = Car on drivableRoad,
#         facing offset relative to roadDirection,
#         with visibleDistance 50,
#         with viewAngle 135 deg

# otherCar = Car on visible road,
#             facing Range(-15, 15) deg relative to roadDirection

# require (distance from ego to otherCar) < 10

# %%
### Scenic Code ###
# ego = Car on drivableRoad,
#         facing Range(-15, 15) deg relative to roadDirection,
#         with visibleDistance 50,
#         with viewAngle 135 deg

# other1 = Car on intersection,
#             facing -1 * Range(50, 135) deg relative to ego.heading

# other2 = Car on intersection,
#             facing -1 * Range(50, 135) deg relative to ego.heading

# require abs(relative heading of other1 from other2) > 100 deg
# require (distance from ego to intersectionRegion) < 10

# %%
### Scenic Code ###
# ego = Car on drivableRoad,
#         facing Range(-15, 15) deg relative to roadDirection,
#         with visibleDistance 50,
#         with viewAngle 135 deg

# point1 = OrientedPoint ahead of ego by Range(0, 40)
# Car at (point1 offset by Range(-1, 1) & 0),
#     facing Range(-15, 15) deg relative to roadDirection

# oppositeCar = Car offset by (Range(-10, -1), Range(0, 50)),
#     facing Range(140, 180) deg relative to ego.heading

# point2 = OrientedPoint ahead of oppositeCar by Range(0, 40)
# Car at (point2 offset by Range(-1, 1) @ 0),
#     facing Range(-15, 15) deg relative to roadDirection

# %%
### Scenic Code ###
# lanesWithRightLane = filter(lambda i: i._laneToRight, network.laneSections)
# egoLane = Uniform(*lanesWithRightLane)

# ego = Car on egoLane,
#         facing Range(-15, 15) deg relative to roadDirection
# cutInCar = Car offset by Range(0, 4) @ Range(0, 5),
#             facing -1*Range(15, 30) deg relative to roadDirection

# %%
# I think there are 3 main things that need to now be implemented in Apperception to allow incorparation with Scenic:
# 1. A way to have the arbitrary filters that were possible in the old API (the predicate lambda functions that could be passed)
# 2. A way to have filters with regardes to other objects. For example, I could say I want "cars that are to the left of a bus by 0.25m" or such. I would assume this could also be implemented as a lambda function filter (I have included an example fo this in the scenic_equivelants notebook).
# 3. Some way to not only recognize what the type of an object is, but recognize the type of point it is on. For example, recognizing that the Car is on a road, or that the Car is in an intersection (this is something that is done quite a lot in Scenic).
#      - For this, we might not have to incorporate it into apperception, and can make it the users responsibility (and they can create their own filters that do this), but I am not too sure

# %%



