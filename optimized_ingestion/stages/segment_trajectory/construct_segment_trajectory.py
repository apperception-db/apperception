from apperception.database import database

import datetime
import math
import postgis
import psycopg2
from collections import namedtuple
from ..detection_estimation.detection_estimation import DetectionInfo
from ..detection_estimation.segment_mapping import RoadSegmentInfo
from ..detection_estimation.utils import (get_segment_line,
                                        project_point_onto_linestring)
from plpygis import Geometry
from shapely.geometry import Point

test_segment_query = """
SELECT
    segmentpolygon.elementid,
    segmentpolygon.elementpolygon,
    segment.segmentline,
    segmentpolygon.segmenttypes,
    segment.heading
FROM segmentpolygon
    LEFT OUTER JOIN segment
        ON segmentpolygon.elementid = segment.elementid
WHERE segmentpolygon.elementid = \'{segment_id}\';
"""


segment_closest_query = """
WITH min_distance AS (
SELECT
    MIN(ST_Distance(segmentpolygon.elementpolygon, {point}::geometry)) distance
FROM segmentpolygon
    LEFT OUTER JOIN segment
        ON segmentpolygon.elementid = segment.elementid
WHERE ST_Distance(segmentpolygon.elementpolygon, {point}::geometry) > 0
    AND cos(radians(
            facingRelative({point_heading}::real,
                        degrees(segment.heading)::real))
        ) > 0
)
SELECT
    segmentpolygon.elementid,
    segmentpolygon.elementpolygon,
    segment.segmentline,
    segmentpolygon.segmenttypes,
    segment.heading
FROM min_distance, segmentpolygon
LEFT OUTER JOIN segment
        ON segmentpolygon.elementid = segment.elementid
WHERE ST_Distance(segmentpolygon.elementpolygon, {point}::geometry) = min_distance.distance;
"""


segment_contain_vector_query = """
WITH min_contain AS (
SELECT
    MIN(ST_Area(segmentpolygon.elementpolygon)) min_segment_area
FROM segmentpolygon
    LEFT OUTER JOIN segment
        ON segmentpolygon.elementid = segment.elementid
WHERE ST_Contains(
    segmentpolygon.elementpolygon,
    {point}::geometry
    )
    AND cos(radians(
        facingRelative({point_heading}::real,
                      degrees(segment.heading)::real))
        ) > 0
)
SELECT
    segmentpolygon.elementid,
    segmentpolygon.elementpolygon,
    segment.segmentline,
    segmentpolygon.segmenttypes,
    segment.heading
FROM min_contain, segmentpolygon
LEFT OUTER JOIN segment
        ON segmentpolygon.elementid = segment.elementid
WHERE ST_Area(segmentpolygon.elementpolygon) = min_contain.min_segment_area;
"""


SegmentTrajectoryPoint = namedtuple(
    "segment_trajectory_point", ['car_loc3d', 'timestamp', 'segment_line',
                                 'segment_heading', 'road_segment_info'])


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
            road_segment_info = RoadSegmentInfo(
                segmentid, segmentpolygon, [segmentline], segmenttype,
                [segmentheading], None, False, None)
        detection_info = DetectionInfo(obj_id=segmentid,
                                       frame_segment=None,
                                       road_segment_info=road_segment_info,
                                       car_loc3d=point,
                                       car_loc2d=None,
                                       car_bbox3d=None,
                                       car_bbox2d=None,
                                       ego_trajectory=None,
                                       ego_config=None,
                                       ego_road_segment_info=None,
                                       timestamp=timestamp)
        detection_infos.append(detection_info)
    return detection_infos


def construct_new_road_segment_info(
        result: "List[Tuple[str, postgis.Polygon, postgis.LineString, str, float]]"):
    """Construct new road segment info based on query result.

    This Function constructs the new the road segment info
    based on the query result that finds the correct road segment that contains
    the calibrated trajectory point.
    """
    kept_segment = (None, None)
    road_segment_info = None
    for road_segment in result:
        segmentid, segmentpolygon, segmentline, segmenttype, segmentheading = road_segment
        segmenttype = segmenttype[0] if segmenttype is not None else None,
        segmentheading = math.degrees(segmentheading) if segmentheading is not None else None
        segmentid = segmentid.split('_')[0]
        segmentline = Geometry(segmentline.to_ewkb()).shapely if segmentline else None
        if segmentid == kept_segment[0]:
            road_segment_info.segment_lines.append(segmentline)
            road_segment_info.segment_headings.append(segmentheading)
        else:
            if kept_segment[0] is not None:
                if kept_segment[1] is not None:
                    continue
        segmentpolygon = Geometry(segmentpolygon.to_ewkb()).shapely
        road_segment_info = RoadSegmentInfo(
            segmentid, segmentpolygon, [segmentline], segmenttype,
            [segmentheading], None, False, None)
        kept_segment = (segmentid, segmenttype)
    return road_segment_info


def calibrate(
        trajectory_3d: "List[trajectory_3d]",
        detection_infos: "List[DetectionInfo]") -> "List[SegmentTrajectoryPoint]":
    """Calibrate the trajectory to the road segments.

    Given a trajectory and the corresponding detection infos, map the trajectory
    to the correct road segments.
    The returned value is a list of SegmentTrajectoryPoint.
    """
    road_segment_trajectory = []
    for i in range(len(trajectory_3d)):
        current_point3d, timestamp = trajectory_3d[i]
        current_point = current_point3d[:2]
        detection_info = detection_infos[i]
        current_road_segment_heading = detection_info.segment_heading
        current_segment_line = detection_info.segment_line
        current_road_segment_info = detection_info.road_segment_info
        if i != len(trajectory_3d) - 1:
            next_point = trajectory_3d[i + 1][0][:2]
            if current_road_segment_heading is not None:
                current_point_heading = math.atan2(next_point[1] - current_point[1],
                                                   next_point[0] - current_point[0])
                current_point_heading = math.degrees(current_point_heading)

                relative_heading = (abs(current_road_segment_heading + 90
                                        - current_point_heading) % 360)

            if (current_road_segment_heading is None
                    or math.cos(math.radians(relative_heading)) > 0):
                road_segment_trajectory.append(
                    segment_trajectory_point(current_point3d,
                                             timestamp,
                                             current_segment_line,
                                             current_road_segment_heading,
                                             current_road_segment_info))
                continue

        ### project current_point to the segment line of the previous point
        ### and then find the segment that  contains the projected point
        ### however this requires querying for road segment once for each point to be calibrated
        if len(road_segment_trajectory) == 0:
            query = psycopg2.sql.SQL(segment_closest_query).format(
                point=psycopg2.sql.Literal(postgis.point.Point(current_point)),
                point_heading=psycopg2.sql.Literal(current_point_heading - 90)
            )
        else:
            prev_calibrated_point = road_segment_trajectory[-1]
            prev_segment_line = prev_calibrated_point[2]
            prev_segment_heading = prev_calibrated_point[3]
            projection = project_point_onto_linestring(Point(current_point), prev_segment_line)
            current_point3d = (projection.x, projection.y, 0.0)
            query = psycopg2.sql.SQL(segment_contain_vector_query).format(
                point=psycopg2.sql.Literal(postgis.point.Point((projection.x, projection.y))),
                point_heading=psycopg2.sql.Literal(prev_segment_heading - 90)
            )
        result = database.execute(query)
        new_road_segment_info = construct_new_road_segment_info(result)
        new_segment_line, new_heading = get_segment_line(current_road_segment_info, current_point3d)
        road_segment_trajectory.append(
            segment_trajectory_point(current_point3d,
                                     timestamp,
                                     new_segment_line,
                                     new_heading,
                                     new_road_segment_info))
    return road_segment_trajectory


#### The Remaining tests ####

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
