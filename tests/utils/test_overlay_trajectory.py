from apperception.world import empty_world
# from apperception.utils import overlay_trajectory
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

    data_dir =  "data/scenic/images"

    world.overlay_trajectory(images_data_path=data_dir, overlay_headings=True, overlay_objects=True, overlay_road=True, keep_whole_video=True)