from apperception.database import database

import datetime
import math
from plpygis import Geometry

from .construct_segment_trajectory import calibrate, test_segment_query
from ..detection_estimation.detection_estimation import DetectionInfo
from ..detection_estimation.segment_mapping import RoadPolygonInfo


def get_test_trajectory(test_trajectory_points):
    """This function is for testing only."""
    start_time = datetime.datetime.now()
    trajectory = []
    for i in range(len(test_trajectory_points)):
        start_time += datetime.timedelta(seconds=5)
        trajectory.append((test_trajectory_points[i], start_time))
    return trajectory


def get_test_detection_infos(test_trajectory, test_segments):
    """This function is for testing only."""
    assert len(test_trajectory) == len(test_segments)
    detection_infos = []
    for i in range(len(test_trajectory)):
        point, timestamp = test_trajectory[i]
        test_segments_for_current_point = test_segments[i]
        road_segment_info = None
        for test_segment in test_segments_for_current_point:
            segmentid, segmentpolygon, segmentline, segmenttype, segmentheading = test_segment
            segmentheading = math.degrees(segmentheading) if segmentheading is not None else None
            segmentline = Geometry(segmentline.to_ewkb()).shapely if segmentline else None
            if road_segment_info is not None:
                road_segment_info.segment_lines.append(segmentline)
                road_segment_info.segment_headings.append(segmentheading)
                continue
            segmentpolygon = Geometry(segmentpolygon.to_ewkb()).shapely
            road_segment_info = RoadPolygonInfo(
                segmentid, segmentpolygon, [segmentline], segmenttype,
                [segmentheading], None, False, None)
        detection_info = DetectionInfo(detection_id=segmentid,
                                       frame_segment=None,
                                       road_polygon_info=road_segment_info,
                                       car_loc3d=point,
                                       car_loc2d=None,
                                       car_bbox3d=None,
                                       car_bbox2d=None,
                                       ego_trajectory=None,
                                       ego_config=None,
                                       ego_road_polygon_info=None,
                                       timestamp=timestamp)
        detection_infos.append(detection_info)
    return detection_infos


def test_same_segment():
    print("test same segment")
    test_segment_ids = ['99c90907-e7a2-4b19-becc-afe2b7f013c7',
                        'c67e592f-2e73-4165-b8cf-64165bb300a8',
                        '99c90907-e7a2-4b19-becc-afe2b7f013c7',
                        'c67e592f-2e73-4165-b8cf-64165bb300a8',]
    test_segments = [database.execute(test_segment_query.format(segment_id=segment_id))
                     for segment_id in test_segment_ids]
    test_trajectory_points = [(1955, 870), (1960, 874), (1980, 872), (1990, 875)]
    test_trajectory = get_test_trajectory(test_trajectory_points)
    test_detection_infos = get_test_detection_infos(test_trajectory, test_segments)
    segment_trajectory = calibrate(test_trajectory, test_detection_infos)
    correct_result = ['99c90907-e7a2-4b19-becc-afe2b7f013c7',
                      '53f56897-4795-4d75-a721-3c969bb3206c',
                      '99c90907-e7a2-4b19-becc-afe2b7f013c7',
                      '99c90907-e7a2-4b19-becc-afe2b7f013c7']
    print([segment_trajectory_point.road_segment_info.segment_id
           for segment_trajectory_point in segment_trajectory])
    print("correct result", correct_result)


def test_wrong_start_same_segment():
    print("test wrong start same segment")
    test_segment_ids = ['c67e592f-2e73-4165-b8cf-64165bb300a8',
                        '99c90907-e7a2-4b19-becc-afe2b7f013c7',
                        '99c90907-e7a2-4b19-becc-afe2b7f013c7',
                        'c67e592f-2e73-4165-b8cf-64165bb300a8',]
    test_segments = [database.execute(test_segment_query.format(segment_id=segment_id))
                     for segment_id in test_segment_ids]
    test_trajectory_points = [(1955, 874), (1960, 870), (1980, 872), (1990, 875)]
    test_trajectory = get_test_trajectory(test_trajectory_points)
    test_detection_infos = get_test_detection_infos(test_trajectory, test_segments)
    segment_trajectory = calibrate(test_trajectory, test_detection_infos)
    correct_result = ['99c90907-e7a2-4b19-becc-afe2b7f013c7',
                      '99c90907-e7a2-4b19-becc-afe2b7f013c7',
                      '99c90907-e7a2-4b19-becc-afe2b7f013c7',
                      '99c90907-e7a2-4b19-becc-afe2b7f013c7']
    print([segment_trajectory_point.road_segment_info.segment_id
           for segment_trajectory_point in segment_trajectory])
    print("correct result", correct_result)


def test_connected_segments():
    print("test connected segment")
    """Some trajectory points are in the wrong segments."""
    test_segment_ids = ['34c01bd5-f649-42e2-be32-30f9a4d02b25',
                        'e39e4059-3a55-42f9-896f-475d89a70e86',
                        '34c01bd5-f649-42e2-be32-30f9a4d02b25',
                        'aa22ee59-c9ef-4759-a69c-c295469f3e37_inter',
                        '99c90907-e7a2-4b19-becc-afe2b7f013c7',
                        'c67e592f-2e73-4165-b8cf-64165bb300a8',
                        '99c90907-e7a2-4b19-becc-afe2b7f013c7',
                        'c67e592f-2e73-4165-b8cf-64165bb300a8',]
    test_segments = [database.execute(test_segment_query.format(segment_id=segment_id))
                     for segment_id in test_segment_ids]
    test_trajectory_points = [(1910, 869), (1915, 873), (1920, 871), (1940, 870),
                              (1955, 870), (1960, 874), (1980, 872), (1990, 875),]
    test_trajectory = get_test_trajectory(test_trajectory_points)
    test_detection_infos = get_test_detection_infos(test_trajectory, test_segments)
    segment_trajectory = calibrate(test_trajectory, test_detection_infos)
    correct_result = ['34c01bd5-f649-42e2-be32-30f9a4d02b25',
                      '2e6d0881-bb10-4145-a45f-28382c46e476',
                      '34c01bd5-f649-42e2-be32-30f9a4d02b25',
                      'aa22ee59-c9ef-4759-a69c-c295469f3e37_inter',
                      '99c90907-e7a2-4b19-becc-afe2b7f013c7',
                      '53f56897-4795-4d75-a721-3c969bb3206c',
                      '99c90907-e7a2-4b19-becc-afe2b7f013c7',
                      '99c90907-e7a2-4b19-becc-afe2b7f013c7']
    print([segment_trajectory_point.road_segment_info.segment_id
           for segment_trajectory_point in segment_trajectory])
    print("correct result", correct_result)


def test_complete_story1():
    """Simplest complete story case.

    The trajectories are all in the correct segments.
    """
    print("test complete story 1")
    test_segment_ids = ['34c01bd5-f649-42e2-be32-30f9a4d02b25',
                        '34c01bd5-f649-42e2-be32-30f9a4d02b25',
                        '99c90907-e7a2-4b19-becc-afe2b7f013c7',
                        '99c90907-e7a2-4b19-becc-afe2b7f013c7',
                        '99c90907-e7a2-4b19-becc-afe2b7f013c7',
                        '99c90907-e7a2-4b19-becc-afe2b7f013c7',]
    test_segments = [database.execute(test_segment_query.format(segment_id=segment_id))
                     for segment_id in test_segment_ids]
    test_trajectory_points = [(1910, 869), (1920, 871), (1955, 870), (1960, 871),
                              (1980, 871), (1990, 871),]
    test_trajectory = get_test_trajectory(test_trajectory_points)
    test_detection_infos = get_test_detection_infos(test_trajectory, test_segments)
    segment_trajectory = calibrate(test_trajectory, test_detection_infos)
    correct_result = ['34c01bd5-f649-42e2-be32-30f9a4d02b25',
                      '34c01bd5-f649-42e2-be32-30f9a4d02b25',
                      'aa22ee59-c9ef-4759-a69c-c295469f3e37_inter',
                      '99c90907-e7a2-4b19-becc-afe2b7f013c7',
                      '99c90907-e7a2-4b19-becc-afe2b7f013c7',
                      '99c90907-e7a2-4b19-becc-afe2b7f013c7',
                      '99c90907-e7a2-4b19-becc-afe2b7f013c7',]
    print([segment_trajectory_point.road_segment_info.segment_id
           for segment_trajectory_point in segment_trajectory])
    print("correct result", correct_result)


def test_complete_story2():
    """Some trajectory points are in the wrong segments."""
    print("test complete story 2")
    test_segment_ids = ['34c01bd5-f649-42e2-be32-30f9a4d02b25',
                        'e39e4059-3a55-42f9-896f-475d89a70e86',
                        '99c90907-e7a2-4b19-becc-afe2b7f013c7',
                        'c67e592f-2e73-4165-b8cf-64165bb300a8',
                        '99c90907-e7a2-4b19-becc-afe2b7f013c7',
                        'c67e592f-2e73-4165-b8cf-64165bb300a8',]
    test_segments = [database.execute(test_segment_query.format(segment_id=segment_id))
                     for segment_id in test_segment_ids]
    test_trajectory_points = [(1910, 869), (1920, 874), (1955, 870),
                              (1960, 874), (1980, 872), (1990, 875),]
    test_trajectory = get_test_trajectory(test_trajectory_points)
    test_detection_infos = get_test_detection_infos(test_trajectory, test_segments)
    segment_trajectory = calibrate(test_trajectory, test_detection_infos)
    correct_result = ['34c01bd5-f649-42e2-be32-30f9a4d02b25',
                      '34c01bd5-f649-42e2-be32-30f9a4d02b25',
                      'aa22ee59-c9ef-4759-a69c-c295469f3e37_inter',
                      '99c90907-e7a2-4b19-becc-afe2b7f013c7',
                      '99c90907-e7a2-4b19-becc-afe2b7f013c7',
                      '99c90907-e7a2-4b19-becc-afe2b7f013c7',
                      '99c90907-e7a2-4b19-becc-afe2b7f013c7',]
    print([segment_trajectory_point.road_segment_info.segment_id
           for segment_trajectory_point in segment_trajectory])
    print("correct result", correct_result)


def test_complete_story3():
    """Some trajectory points are in the wrong segments."""
    print("test complete story 3")
    test_segment_ids = ['34c01bd5-f649-42e2-be32-30f9a4d02b25',
                        'e39e4059-3a55-42f9-896f-475d89a70e86',
                        '34c01bd5-f649-42e2-be32-30f9a4d02b25',
                        '9eef6c56-c5d9-46ed-a44e-9848676bdddf',
                        '53c5901a-dad9-4f0d-bcb6-c127dda2be09',
                        '9eef6c56-c5d9-46ed-a44e-9848676bdddf',
                        '53c5901a-dad9-4f0d-bcb6-c127dda2be09',]
    test_segments = [database.execute(test_segment_query.format(segment_id=segment_id))
                     for segment_id in test_segment_ids]
    test_trajectory_points = [(1910, 869), (1915, 873), (1920, 871),
                              (1937, 882), (1932, 885), (1937, 887), (1932, 892),]
    test_trajectory = get_test_trajectory(test_trajectory_points)
    test_detection_infos = get_test_detection_infos(test_trajectory, test_segments)
    segment_trajectory = calibrate(test_trajectory, test_detection_infos)
    correct_result = ['34c01bd5-f649-42e2-be32-30f9a4d02b25',
                      '34c01bd5-f649-42e2-be32-30f9a4d02b25',
                      '34c01bd5-f649-42e2-be32-30f9a4d02b25',
                      'aa22ee59-c9ef-4759-a69c-c295469f3e37_inter',
                      '9eef6c56-c5d9-46ed-a44e-9848676bdddf',
                      '9eef6c56-c5d9-46ed-a44e-9848676bdddf',
                      '9eef6c56-c5d9-46ed-a44e-9848676bdddf',
                      '9eef6c56-c5d9-46ed-a44e-9848676bdddf',]
    print([segment_trajectory_point.road_segment_info.segment_id
           for segment_trajectory_point in segment_trajectory])
    print("correct result", correct_result)


def test_complete_story4():
    """Most trajectory points are in the wrong segments."""
    print("test complete story 4")
    test_segment_ids = ['34c01bd5-f649-42e2-be32-30f9a4d02b25',
                        'e39e4059-3a55-42f9-896f-475d89a70e86',
                        '53c5901a-dad9-4f0d-bcb6-c127dda2be09',
                        '9eef6c56-c5d9-46ed-a44e-9848676bdddf',
                        '53c5901a-dad9-4f0d-bcb6-c127dda2be09',]
    test_segments = [database.execute(test_segment_query.format(segment_id=segment_id))
                     for segment_id in test_segment_ids]
    test_trajectory_points = [(1910, 868), (1920, 873), (1932, 885), (1937, 887), (1932, 892),]
    test_trajectory = get_test_trajectory(test_trajectory_points)
    test_detection_infos = get_test_detection_infos(test_trajectory, test_segments)
    segment_trajectory = calibrate(test_trajectory, test_detection_infos)
    correct_result = ['34c01bd5-f649-42e2-be32-30f9a4d02b25',
                      '34c01bd5-f649-42e2-be32-30f9a4d02b25',
                      'aa22ee59-c9ef-4759-a69c-c295469f3e37_inter',
                      '9eef6c56-c5d9-46ed-a44e-9848676bdddf',
                      '9eef6c56-c5d9-46ed-a44e-9848676bdddf',
                      '9eef6c56-c5d9-46ed-a44e-9848676bdddf',]
    print([segment_trajectory_point.road_segment_info.segment_id
           for segment_trajectory_point in segment_trajectory])
    print("correct result", correct_result)


if __name__ == '__main__':
    test_same_segment()
    test_wrong_start_same_segment()
    test_connected_segments()
    test_complete_story1()
    test_complete_story2()
    test_complete_story3()
    test_complete_story4()
    print('All tests passed!')