import time
from os import environ

from apperception.database import database
from apperception.world import empty_world
from apperception.utils import F
from apperception.predicate import camera, objects, lit
database.connection

name = 'ScenicWorld' # world name
world = empty_world(name=name)

obj1 = objects[0]
cam = camera
world = world.filter(
    (F.like(obj1.type, 'car') | F.like(obj1.type, 'truck') | F.like(obj1.type, 'bus')) &
    (F.distance(cam.ego, obj1.trans@cam.timestamp) < 50) &
    (F.view_angle(obj1.trans@cam.time, cam.ego) < 70 / 2) &
    F.angle_between(F.facing_relative(cam.ego, F.road_direction(cam.ego, cam.ego)), -180, -90) &
    F.contained(cam.ego, F.road_segment('road')) &
    F.contained(obj1.trans@cam.time, F.road_segment('road')) &
    F.angle_between(F.facing_relative(obj1.trans@cam.time, F.road_direction(obj1.traj@cam.time, obj1.trans@cam.time)), -15, 15) &
    (F.distance(cam.ego, obj1.trans@cam.time) < 10)
)


start = time.time()

id_time_camId_filename = world.get_id_time_camId_filename(1)

end = time.time()
print("Figure 14 (baseline): ", format(end-start))

filenames = list(set([x[3] for x in id_time_camId_filename if "samples" in x[3]]))
with open('fig14_baseline_results.txt', 'w') as f:
    for line in filenames:
        f.write(f"{line}\n")