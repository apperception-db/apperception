from apperception.world import empty_world
from datetime import datetime, timezone


def test_fig_13():
    world = empty_world(name='world')
    world = world.filter("lambda obj1, cam: " +
        "F.like(obj1.object_type, 'vehicle%') and " +
        "F.distance(cam.ego, obj1, cam.timestamp) < 50 and " +
        "F.view_angle(obj1, cam.ego, cam.timestamp) < 70 / 2 and " +
        "F.angle_between(F.facing_relative(cam.ego, F.road_direction(cam.ego, cam.timestamp, cam.ego), cam.timestamp), -180, -90) and " +
        "F.contained(cam.ego, F.road_segment('road')) and " +
        "F.contained(obj1.traj, F.road_segment('road'), cam.timestamp) and " +
        "F.angle_between(F.facing_relative(obj1, F.road_direction(obj1.traj, cam.timestamp, cam.ego), cam.timestamp), -15, 15) and " +
        "F.distance(cam.ego, obj1, cam.timestamp) < 10"
    )

    assert set(world.get_id_time_camId_filename(2)) == set([
    ])