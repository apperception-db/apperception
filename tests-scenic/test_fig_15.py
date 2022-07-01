from apperception.world import empty_world
from datetime import datetime, timezone


def test_fig_13():
    world = empty_world(name='world')
    world = world.filter(" ".join([
        "lambda car1, opposite_car, car2, cam:",
        "F.like(car1.object_type, 'vehicle%') and",
        "F.angle_between(F.facing_relative(cam.ego, F.road_direction(cam.ego, cam.timestamp, cam.ego), cam.timestamp), -15, 15) and",
        "F.view_angle(car1, cam.ego, cam.timestamp) < 70 / 2 and",
        "F.distance(cam.ego, car1, cam.timestamp) < 40 and",
        "F.angle_between(F.facing_relative(car1, cam.ego, cam.timestamp), -15, 15) and",
        "F.angle_between(F.facing_relative(car1, F.road_direction(car1.traj, cam.timestamp, cam.ego), cam.timestamp), -15, 15) and",
        "F.ahead(car1, cam.ego, cam.timestamp) and",
        "F.angle_between(F.facing_relative(cam.ego, F.road_direction(cam.ego, cam.timestamp, cam.ego), cam.timestamp), -15, 15) and",
        "F.convert_camera(opposite_car, cam.ego, cam.timestamp) > [-10, 0] and",
        "F.convert_camera(opposite_car, cam.ego, cam.timestamp) < [-1, 50] and",
        "F.angle_between(F.facing_relative(opposite_car, cam.ego, cam.timestamp), 140, 180) and",
        "F.like(car2.object_type, 'vehicle%') and F.like(opposite_car.object_type, 'vehicle%') and",
        "opposite_car.itemId != car2.itemId and car1.itemId != car2.itemId and car1.itemId != opposite_car.itemId and",
        "F.distance(opposite_car, car2, cam.timestamp) < 40 and",
        "F.angle_between(F.facing_relative(car2, F.road_direction(car2.traj, cam.timestamp, cam.ego), cam.timestamp), -15, 15) and",
        "F.ahead(car2, opposite_car, cam.timestamp)"
    ]))

    assert world.get_id_time_camId_filename(1) == []
