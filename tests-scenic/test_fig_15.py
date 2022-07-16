from apperception.world import empty_world
from datetime import datetime, timezone


def test_fig_15():
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

    assert world.get_id_time_camId_filename(1) == [
        (
            '6a81ab78eee3477e8509569a5d0a2217',
            datetime(2018, 7, 26, 9, 18, 40, 162404, tzinfo=timezone.utc),
            'scene-0207',
            'samples/CAM_FRONT/n008-2018-07-26-12-13-50-0400__CAM_FRONT__1532621920162404.jpg'
        ),
        (
            '6a81ab78eee3477e8509569a5d0a2217',
            datetime(2018, 7, 26, 9, 18, 40, 662404, tzinfo=timezone.utc),
            'scene-0207',
            'samples/CAM_FRONT/n008-2018-07-26-12-13-50-0400__CAM_FRONT__1532621920662404.jpg'
        ),
    ]
