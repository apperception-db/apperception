from apperception.world import empty_world


def test_fig_16():
    world = empty_world(name='world')
    world = world.filter(" ".join([
        "lambda obj1, cam:",
        "F.contained(cam.ego, F.road_segment('lanewithrightlane')) and",
        "F.angle_between(F.facing_relative(cam.ego, F.road_direction(cam.ego, cam.timestamp, cam.ego), cam.timestamp), -15, 15) and",
        "F.like(obj1.object_type, 'vehicle%') and",
        "F.convert_camera(obj1, cam.ego, cam.timestamp) > [0,0] and",
        "F.convert_camera(obj1, cam.ego, cam.timestamp) < [4,5] and",
        "F.angle_between(F.facing_relative(obj1, F.road_direction(obj1.traj, cam.timestamp, cam.ego), cam.timestamp), -30, -15)",
    ]))

    assert world.get_id_time_camId_filename(1) == []
