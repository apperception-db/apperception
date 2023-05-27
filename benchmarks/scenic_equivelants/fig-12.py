import time
from os import environ

from apperception.database import database
from apperception.world import empty_world
from apperception.utils import F
from apperception.predicate import camera, objects, lit
database.connection

name = 'ScenicWorld' # world name
world = empty_world(name=name)

o = objects[0]
c = camera
world = world.filter(
    (F.like(c.filename, 'samples/CAM_FRONT/%')) &  ########
    F.like(o.type, lit('human.pedestrian%')) & ########
    F.contained(c.ego, 'road') & ########
    (F.contained(o.trans@c.time, 'road')) & ########
    F.angle_excluding(F.facing_relative(o.traj@c.time, c.ego), lit(-70), lit(70)) & ########
    F.angle_between(F.facing_relative(c.ego,  F.road_direction(c.ego)), lit(-15), lit(15)) & ########
    (F.distance(c.ego, o.trans@c.time) < lit(50)) & ########
    (F.view_angle(o.trans@c.time, c.ego) < lit(35)) ########
)

start = time.time()
# keys = world.get_traj_key()
id_time_camId_filename = world.get_id_time_camId_filename(num_joined_tables=1)

end = time.time()
print("Figure 12 (baseline): ", format(end-start))

filenames = list(set([x[3] for x in id_time_camId_filename if "samples" in x[3]]))
with open('fig12_baseline_results.txt', 'w') as f:
    for line in filenames:
        f.write(f"{line}\n")