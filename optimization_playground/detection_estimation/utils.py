from collections import namedtuple
import datetime
import math

from apperception.database import database
from apperception.utils import F, transformation, fetch_camera_config, fetch_camera_trajectory
from shapely.geometry import Point, Polygon, LineString, MultiLineString, box


SAME_DIRECTION = 'same_direction'
OPPOSITE_DIRECTION = 'opposite_direction'
trajectory_3d = namedtuple('trajectory_3d', ['coordinates', 'timestamp'])
temporal_speed = namedtuple('temporal_speed', ['speed', 'timestamp'])

def mph_to_mps(mph):
    return mph * 0.44704
MAX_CAR_SPEED = {
    'lane': 35,
    'road': 35,
    'laneSection': 35,
    'roadSection': 35,
    'intersection': 25,
    'highway': 55,
    'residential': 25,
}
MAX_CAR_SPEED.update({k: mph_to_mps(v) for k, v in MAX_CAR_SPEED.items()})

def time_elapse(current_time, elapsed_time):
    return current_time + datetime.timedelta(seconds=elapsed_time)

def compute_area(polygon):
    return box(*polygon).area

def compute_distance(loc1, loc2):
    return Point(loc1).distance(Point(loc2))

def relative_direction(vec1, vec2):
    return (vec1[0]*vec2[0] + vec1[1]*vec2[1])/math.sqrt(vec1[0]**2 + vec1[1]**2)/math.sqrt(vec2[0]**2 + vec2[1]**2) > 0

def _construct_extended_line(polygon, line):
    polygon = Polygon(polygon)
    line = LineString(line)
    minx, miny, maxx, maxy = polygon.bounds
    bounding_box = box(minx, miny, maxx, maxy)
    a, b = line.boundary
    if a.x == b.x:  # vertical line
        extended_line = LineString([(a.x, miny), (a.x, maxy)])
    elif a.y == b.y:  # horizonthal line
        extended_line = LineString([(minx, a.y), (maxx, a.y)])
    else:
        # linear equation: y = k*x + m
        k = (b.y - a.y) / (b.x - a.x)
        m = a.y - k * a.x
        y0 = k * minx + m
        y1 = k * maxx + m
        x0 = (miny - m) / k
        x1 = (maxy - m) / k
        points_on_boundary_lines = [Point(minx, y0), Point(maxx, y1), 
                                    Point(x0, miny), Point(x1, maxy)]
        points_sorted_by_distance = sorted(points_on_boundary_lines, key=bounding_box.distance)
        extended_line = LineString(points_sorted_by_distance[:2])
    return extended_line

def intersection_between_line_and_trajectory(line, trajectory):
    trajectory_to_polygon = Polygon(trajectory)
    extended_line = _construct_extended_line(trajectory, line)
    intersection = extended_line.intersection(LineString(trajectory))
    if not isinstance(intersection, LineString) or intersection.is_empty:
        return tuple()
    elif isinstance(intersection, LineString):
        return tuple(intersection.coords)

def line_to_polygon_intersection(polygon, line):
    try:
        extended_line = _construct_extended_line(polygon, line)
        intersection = extended_line.intersection(polygon)
    except:
        return tuple()
    if intersection.is_empty:
        return tuple()
    elif isinstance(intersection, LineString):
        return tuple(intersection.coords)
    elif isinstance(intersection, MultiLineString):
        all_intersections = []
        for intersect in intersection:
            all_intersections.extend(list(intersect.coords))
        return tuple(all_intersections)
    else:
        raise ValueError('Unexpected intersection type')


### ASSUMPTIONS ###
def max_car_speed(road_type):
    """Maximum speed of a car on the given road type

    For example, the maximum speed of a car on a highway is 65mph, 
    and 25mph on a residential road.
    """
    return MAX_CAR_SPEED[road_type]

def min_car_speed(road_type):
    return max_car_speed(road_type) / 2


### HELPER FUNCTIONS ###
def get_ego_trajectory(video, sorted_ego_config=None):
    if sorted_ego_config is None:
        camera_trajectory_config = fetch_camera_trajectory(video, database)
    else:
        camera_trajectory_config = sorted_ego_config
    return [trajectory_3d(config['egoTranslation'], config['timestamp']) for config in camera_trajectory_config]

def get_ego_speed(ego_trajectory):
    point_wise_temporal_speed = []
    for i in range(len(ego_trajectory) - 1):
        x,y,z = ego_trajectory[i].coordinates
        timestamp = ego_trajectory[i].timestamp
        x_next, y_next, z_next = ego_trajectory[i+1].coordinates
        timestamp_next = ego_trajectory[i+1].timestamp
        distance = compute_distance((x,y), (x_next, y_next))
        point_wise_temporal_speed.append(
            temporal_speed(distance/(timestamp_next - timestamp).total_seconds(),
                           timestamp))
    return point_wise_temporal_speed

def get_ego_avg_speed(ego_trajectory):
    point_wise_ego_speed = get_ego_speed(ego_trajectory)
    return sum([speed.speed for speed in point_wise_ego_speed]) / len(point_wise_ego_speed)

def detection_to_img_segment(car_loc2d, cam_segment_mapping, ego=False):
    maximum_mapping = None
    maximum_mapping_area = 0
    for mapping in cam_segment_mapping:
        cam_segment, road_segment_info = mapping
        if ego and road_segment_info.contains_ego:
            if Polygon(road_segment_info.segment_polygon).area > maximum_mapping_area:
                maximum_mapping = mapping
                maximum_mapping_area = Polygon(road_segment_info.segment_polygon).area
        elif Polygon(cam_segment).contains(Point(car_loc2d)):
            if Polygon(cam_segment).area > maximum_mapping_area:
                maximum_mapping = mapping
                maximum_mapping_area = Polygon(cam_segment).area
    return maximum_mapping

def time_to_nearest_frame(video, timestamp):
    """Return the frame that is closest to the timestamp
    """
    query = f"""
    With nearest_timestamp as (
        SELECT 
            timestamp, abs(extract(epoch from timestamp-\'{timestamp}\')) as diff
        FROM Cameras
        WHERE fileName LIKE '%{video}%'
        Order By diff
        LIMIT 1
    )
    SELECT
        fileName,
        frameNum,
        cameras.timestamp
    FROM cameras, nearest_timestamp
    WHERE
        fileName LIKE '%{video}%'
        AND cameras.timestamp=nearest_timestamp.timestamp
    """
    return database._execute_query(query)[0]

def timestamp_to_nearest_trajectory(trajectory, timestamp):
    """Return the trajectory point that is closest to the timestamp
    """
    return min(trajectory, 
               key=lambda x: abs((x.timestamp - timestamp).total_seconds()))

def point_to_nearest_trajectory(point, trajectory):
    """Return the trajectory point that is closest to the point
    """
    return min(trajectory,
               key=lambda x: compute_distance(x.coordinates, point))

def time_to_exit_current_segment(current_segment_info, 
                                 current_time, car_loc, car_trajectory=None):
    """Return the time that the car exit the current segment

    Assumption: 
    car heading is the same as road heading
    car drives at max speed if no trajectory is given
    """
    segmentpolygon = current_segment_info.segment_polygon
    if car_trajectory:
        for point in car_trajectory:
            if (point.timestamp > current_time and
               not Polygon(segmentpolygon).contains(Point(point.coordinates))):
                return point.timestamp, point.coordinates
        return time_elapse(current_time, -1), None
    segmentheading = current_segment_info.segment_heading + 90
    car_loc = Point(car_loc)
    car_vector = (car_loc.x + math.cos(math.radians(segmentheading)),
                  car_loc.y + math.sin(math.radians(segmentheading)))
    car_heading_line = LineString([car_loc, car_vector])
    # print('car_heading_vector', car_heading_line)
    intersection = line_to_polygon_intersection(segmentpolygon, car_heading_line)
    # print("mapped polygon", segat intersection
    if len(intersection) == 2:
        intersection_1_vector = (intersection[0][0] - car_loc.x,
                                 intersection[0][1] - car_loc.y)
        relative_direction_1 = relative_direction(car_vector, intersection_1_vector)
        intersection_2_vector = (intersection[1][0] - car_loc.x,
                                 intersection[1][1] - car_loc.y)
        relative_direction_2 = relative_direction(car_vector, intersection_2_vector)
        distance1 = compute_distance(car_loc, intersection[0])
        distance2 = compute_distance(car_loc, intersection[1])
        if relative_direction_1:
            return time_elapse(current_time, distance1 / max_car_speed(current_segment_info.segment_type)), intersection[0]
        elif relative_direction_2:
            return time_elapse(current_time, distance2 / max_car_speed(current_segment_info.segment_type)), intersection[1]
        else:
            print("wrong car moving direction")
            return time_elapse(current_time, -1), None
    return time_elapse(current_time, -1), None

def meetup(car1_loc,
           car2_loc,
           car1_heading,
           car2_heading,
           road_type,
           current_time,
           car1_trajectory=None,
           car2_trajectory=None,
           car1_speed=None,
           car2_speed=None):
    """estimate the meetup point as the middle point between car1's loc and car2's loc

    If both trajectories are given, the meetup point is the point where the two trajectories meets
    If one trajectory is given, use that trajectory and other car's speed and direction
    If none trajectory is given, use both cars' speed, or max speed if no speed is given and cars' direction

    For timestamp, it's an estimation based on the trajectory or speed
    Return: (timestamp, meetup_point)

    Assumptions: 
        car1 and car2 are driving towards each other, not necessarily the opposite direction
        There shouldn't be a point they intersect, otherwise it's a collision
        If no trajectory, or speed is given, car drives at max speed
        TODO: now I've just implemented the case for ego car to meet another detected car
    """
    car1_loc = Point(car1_loc) if isinstance(car1_loc, tuple) else car1_loc
    car2_loc = Point(car2_loc) if isinstance(car2_loc, tuple) else car2_loc
    if car1_trajectory is not None and car2_trajectory is None:
        car2_speed = max_car_speed(road_type) if car2_speed is None else car2_speed
        car2_heading += 90
        car2_vector = (car2_loc.x + math.cos(math.radians(car2_heading)),
                       car2_loc.y + math.sin(math.radians(car2_heading)),)
        car2_heading_line = (car2_loc, car2_vector)
        car1_trajectory_points = [point.coordinates for point in car1_trajectory
                                  if point.timestamp > current_time]
        intersection = intersection_between_line_and_trajectory(
            car2_heading_line, car1_trajectory_points)
        if len(intersection) == 1: # i.e. one car drives towards south, the other towards east
            # print("at intersection 1")
            meetup_point = intersection[0]
            time1 = point_to_nearest_trajectory_timestamp(meetup_point, car1_trajectory)
            distance2 = compute_distance(car2_loc, meetup_point)
            time2 = time_elapse(current_time, distance2 / car2_speed)
            return (min(time1, time2), meetup_point)
        elif len(intersection) == 0: # i.e. one car drives towards south, the other towards north
            # print("at intersection 0")
            meetup_point = Point((car1_loc.x + car2_loc.x) / 2, (car1_loc.y + car2_loc.y) / 2)
            time1 = point_to_nearest_trajectory(meetup_point, car1_trajectory).timestamp
            if time1 < current_time:
                time1 = current_time
            distance2 = compute_distance(car2_loc, meetup_point)
            time2 = time_elapse(current_time, distance2 / car2_speed)
            if time2 < current_time:
                time2 = current_time
            return (min(time1, time2), meetup_point)
            
def catchup_time(car1_loc,
                 car2_loc,
                 road_type=None,
                 car1_trajectory=None,
                 car2_trajectory=None,
                 car1_speed=None, 
                 car2_speed=None):
    """Return the time that car1 catches up to car2
    
    Assumption: 
        1. car1 and car2 are driving towards the same direction
        2. car1 drives at max speed, car2 drives at min speed
           if no trajectory or speed is given
        3. TODO: assume now ego is the slowest, it won't bypass another car
    """
    pass

def in_view(car_loc, ego_loc, view_distance):
    """At this point, we only care about detect cars. So they are in the frame
       in_view means whether the car is recognizable enough
    """
    return compute_distance(car_loc, ego_loc) < view_distance

def time_to_exit_view(ego_loc, car_loc, car_heading, ego_trajectory, current_time, road_type, view_distance):
    """Return the time, and location that the car goes beyond ego's view distance

    Assumption: car drives at max speed
    """
    ego_speed = get_ego_avg_speed(ego_trajectory)
    car_speed = max_car_speed(road_type)
    exit_view_time = time_elapse(current_time, view_distance/(car_speed - ego_speed))
    return timestamp_to_nearest_trajectory(ego_trajectory, exit_view_time)

def relative_direction_to_ego(obj_heading, ego_heading):
    """Return the relative direction to ego
       Now only support opposite and same direction
       TODO: add driving into and driving away from
    """
    if obj_heading is None:
        return None
    relative_heading = abs(obj_heading - ego_heading) % 360
    if math.cos(math.radians(relative_heading)) > 0:
        return SAME_DIRECTION
    else:
        return OPPOSITE_DIRECTION