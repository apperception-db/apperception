import time
from os import environ

from apperception.database import database
from apperception.world import empty_world
from apperception.utils import F
from apperception.predicate import camera, objects, lit
database.connection

name = 'ScenicWorld' # world name
world = empty_world(name=name)

cam = camera
car1 = objects[0]
opposite_car = objects[1]
car2 = objects[2]

world = world.filter(
    (F.like(car1.type, 'car') | F.like(car1.type, 'truck')) &
    (F.like(car2.type, 'car') | F.like(car2.type, 'truck')) &
    (F.like(opposite_car.type, 'car') | F.like(opposite_car.type, 'truck')) &
    (opposite_car.id != car2.id) &
    (car1.id != car2.id) &
    (car1.id != opposite_car.id) &

    F.angle_between(F.facing_relative(cam.ego, F.road_direction(cam.ego, cam.ego)), -15, 15) &
    (F.view_angle(car1.traj@cam.time, cam.ego) < 70 / 2) &
    (F.distance(cam.ego, car1.traj@cam.time) < 50) &
#     F.angle_between(F.facing_relative(car1.traj@cam.time, cam.ego), -15, 15) &
#     F.angle_between(F.facing_relative(car1.traj@cam.time, F.road_direction(car1.traj@cam.time, cam.ego)), -15, 15) &
    F.ahead(car1.traj@cam.time, cam.ego) &
    F.angle_between(F.facing_relative(cam.ego, F.road_direction(cam.ego, cam.ego)), -15, 15) &
    (F.convert_camera(opposite_car.traj@cam.time, cam.ego) > [-10, 0]) &
    (F.convert_camera(opposite_car.traj@cam.time, cam.ego) < [-1, 50]) &
    F.angle_between(F.facing_relative(opposite_car.traj@cam.time, cam.ego), 140, 180) &
    (F.distance(opposite_car@cam.time, car2@cam.time) < 40) &
#     F.angle_between(F.facing_relative(car2.traj@cam.time, F.road_direction(car2.traj@cam.time, cam.ego)), -15, 15) &
    F.ahead(car2.traj@cam.time, opposite_car.traj@cam.time)
)


start = time.time()

id_time_camId_filename = world.get_id_time_camId_filename(3)

end = time.time()
print("Figure 15 (baseline): ", format(end-start))

filenames = list(set([x[5] for x in id_time_camId_filename if "samples" in x[5]]))
with open('fig15_baseline_results.txt', 'w') as f:
    for line in filenames:
        f.write(f"{line}\n")