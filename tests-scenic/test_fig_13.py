from apperception.world import empty_world
from datetime import datetime, timezone


def test_fig_13():
    world = empty_world(name='world')
    world = world.filter(" ".join([
        "lambda obj1, obj2, cam:",
        "obj1.object_id != obj2.object_id and",
        "F.like(obj1.object_type, 'vehicle%') and",
        "F.like(obj2.object_type, 'vehicle%') and",
        "F.angle_between(F.facing_relative(cam.ego, F.road_direction(cam.ego, cam.ego)), -15, 15) and",
        "F.distance(cam.ego, obj1, cam.timestamp) < 50 and",
        "F.view_angle(obj1, cam.ego, cam.timestamp) < 70 / 2.0 and",
        "F.distance(cam.ego, obj2, cam.timestamp) < 50 and",
        "F.view_angle(obj2, cam.ego, cam.timestamp) < 70 / 2.0 and",
        "F.contains_all('intersection', [obj1.traj, obj2.traj]@cam.timestamp) and "
        "F.angle_between(F.facing_relative(obj1, cam.ego, cam.timestamp), 50, 135) and",
        "F.angle_between(F.facing_relative(obj2, cam.ego, cam.timestamp), -135, -50) and",
        "F.minDistance(cam.egoTranslation, F.road_segment('intersection')) < 10 and",
        "F.angle_between(F.facing_relative(obj1, obj2, cam.timestamp), 100, -100)",
    ]))

    assert set(world.get_id_time_camId_filename(2)) == set([
        (
            '9d03c6edb6eb4d49acccb245bdd0c652',
            '65d120d480794b9fbb433dc58512559b',
            datetime(2018, 9, 18, 9, 15, 58, 412404, tzinfo=timezone.utc),
            'scene-0456',
            'samples/CAM_FRONT/n008-2018-09-18-12-07-26-0400__CAM_FRONT__1537287358412404.jpg'
        ),
        (
            '82d680066ddd465dbd3b22fd6a66ed70',
            '58350757f1d04f628aab9b22cf33549b',
            datetime(2018, 8, 30, 12, 25, 18, 112404, tzinfo=timezone.utc),
            'scene-0757',
            'samples/CAM_FRONT/n008-2018-08-30-15-16-55-0400__CAM_FRONT__1535657118112404.jpg'
        ),
        (
            '2c74f27891164c9182d5a0d0102dca8c',
            'eb28d3eeb8ac46b8ac47848a18d41dc5',
            datetime(2018, 8, 30, 12, 25, 27, 112404, tzinfo=timezone.utc),
            'scene-0757',
            'samples/CAM_FRONT/n008-2018-08-30-15-16-55-0400__CAM_FRONT__1535657127112404.jpg'
        ),
        (
            '2c74f27891164c9182d5a0d0102dca8c',
            'eb28d3eeb8ac46b8ac47848a18d41dc5',
            datetime(2018, 8, 30, 12, 25, 27, 612404, tzinfo=timezone.utc),
            'scene-0757',
            'samples/CAM_FRONT/n008-2018-08-30-15-16-55-0400__CAM_FRONT__1535657127612404.jpg'
        ),
        (
            '2c74f27891164c9182d5a0d0102dca8c',
            'eb28d3eeb8ac46b8ac47848a18d41dc5',
            datetime(2018, 8, 30, 12, 25, 28, 112404, tzinfo=timezone.utc),
            'scene-0757',
            'samples/CAM_FRONT/n008-2018-08-30-15-16-55-0400__CAM_FRONT__1535657128112404.jpg'
        ),
        (
            'b327acc1048e44889108740b2304dabc',
            '58350757f1d04f628aab9b22cf33549b',
            datetime(2018, 8, 30, 12, 25, 18, 112404, tzinfo=timezone.utc),
            'scene-0757',
            'samples/CAM_FRONT/n008-2018-08-30-15-16-55-0400__CAM_FRONT__1535657118112404.jpg'
        ),
        (
            'b327acc1048e44889108740b2304dabc',
            '58350757f1d04f628aab9b22cf33549b',
            datetime(2018, 8, 30, 12, 25, 18, 612404, tzinfo=timezone.utc),
            'scene-0757',
            'samples/CAM_FRONT/n008-2018-08-30-15-16-55-0400__CAM_FRONT__1535657118612404.jpg'
        ),
        (
            'b327acc1048e44889108740b2304dabc',
            '58350757f1d04f628aab9b22cf33549b',
            datetime(2018, 8, 30, 12, 25, 19, 112404, tzinfo=timezone.utc),
            'scene-0757',
            'samples/CAM_FRONT/n008-2018-08-30-15-16-55-0400__CAM_FRONT__1535657119112404.jpg'
        ),
    ])