from apperception.world import empty_world
from datetime import datetime, timezone


def test_fig_14():
    world = empty_world(name='world')
    world = world.filter(" ".join([
        "lambda obj1, cam:",
        "F.like(obj1.object_type, 'vehicle%') and",
        "F.distance(cam.ego, obj1, cam.timestamp) < 50 and",
        "F.view_angle(obj1, cam.ego, cam.timestamp) < 70 / 2 and",
        "(F.angle_between(F.facing_relative(cam.ego, F.road_direction(cam.ego, cam.timestamp, cam.ego), cam.timestamp), -180, -90) or F.angle_between(F.facing_relative(cam.ego, F.road_direction(cam.ego, cam.timestamp, cam.ego), cam.timestamp), 90, 180)) and",
        "F.contained(cam.ego, F.road_segment('road')) and",
        "F.contained(obj1.traj, F.road_segment('road'), cam.timestamp) and",
        "F.angle_between(F.facing_relative(obj1, F.road_direction(obj1.traj, cam.timestamp, cam.ego), cam.timestamp), -15, 15) and",
        "F.distance(cam.ego, obj1, cam.timestamp) < 10"
    ]))

    assert set(world.get_id_time_camId_filename(1)) == set([
        (
            '9e02e0dcb5f04d01a4b8f0559d0e7d95',
            datetime(2018, 8, 30, 12, 31, 31, 612404, tzinfo=timezone.utc),
            'scene-0769',
            'samples/CAM_FRONT/n008-2018-08-30-15-16-55-0400__CAM_FRONT__1535657491612404.jpg'
        ),
        (
            '9e02e0dcb5f04d01a4b8f0559d0e7d95',
            datetime(2018, 8, 30, 12, 31, 32, 112404, tzinfo=timezone.utc),
            'scene-0769',
            'samples/CAM_FRONT/n008-2018-08-30-15-16-55-0400__CAM_FRONT__1535657492112404.jpg'
        ),
    ])
