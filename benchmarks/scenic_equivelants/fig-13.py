import time
from os import environ

from apperception.database import database
from apperception.world import empty_world
from apperception.utils import F
from apperception.predicate import camera, objects, lit
database.connection

name = 'ScenicWorld'
world = empty_world(name=name)

obj1 = objects[0]
obj2 = objects[1]
cam = camera

world = world.filter(
    (F.like(cam.filename, 'samples/CAM_FRONT/%')) & 
    (obj1.id != obj2.id) &
    (F.like(obj1.type, 'vehicle.car') | F.like(obj1.type, 'vehicle.truck')) & ########
    (F.like(obj2.type, 'vehicle.car') | F.like(obj2.type, 'vehicle.truck')) & ########
    F.angle_between(F.facing_relative(cam.ego, F.road_direction(cam.ego)), -15, 15) & ########
    (F.distance(cam.ego, obj1.trans@cam.time) < 50) & #########
    (F.view_angle(obj1.trans@cam.time, cam.ego) < 70 / 2.0) & ########
    (F.distance(cam.ego, obj2.trans@cam.time) < 50) & #########
    (F.view_angle(obj2.trans@cam.time, cam.ego) < 70 / 2.0) & ########
    F.contains_all('intersection', [obj1.trans, obj2.trans]@cam.time) & ########
    F.angle_between(F.facing_relative(obj1.trans@cam.time, cam.ego), 50, 135) & ########
    F.angle_between(F.facing_relative(obj2.trans@cam.time, cam.ego), -135, -50) & ########
    # (F.min_distance(cam.ego, 'intersection') < 10) &
    F.angle_between(F.facing_relative(obj1.trans@cam.time, obj2.trans@cam.time), 100, -100) ####
)


start = time.time()

id_time_camId_filename = world.get_id_time_camId_filename(2)

end = time.time()
print("Figure 13 (baseline): ", format(end-start))

filenames = list(set([x[4] for x in id_time_camId_filename if "samples" in x[4]]))
with open('fig13_baseline_results.txt', 'w') as f:
    for line in filenames:
        f.write(f"{line}\n")