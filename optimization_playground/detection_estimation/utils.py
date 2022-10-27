from collections import namedtuple

from apperception.database import database
from apperception.utils import F, transformation, fetch_camera_config, fetch_camera_trajectory
from shapely.geometry import Point, Polygon, LineString


SAME_DIRECTION = 'same_direction'
OPPOSITE_DIRECTION = 'opposite_direction'
trajectory_3d = namedtuple('trajectory_3d', ['coordinates', 'timestamp'])

def mph_to_mps(mph):
    return mph * 0.44704
MAX_CAR_SPEED = {
    'lane': 45,
    'intersection': 25,
    'highway': 65,
    'residential': 25,
}
MAX_CAR_SPEED.update({k: mph_to_mps(v) for k, v in MAX_CAR_SPEED.items()})

def compute_distance(loc1, loc2):
    return Point(loc1).distance(Point(loc2))

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
    point_wise_speed = []
    for i in range(len(ego_trajectory) - 1):
        x,y,z = ego_trajectory[i].coordinates
        timestamp = ego_trajectory[i].timestamp
        x_next, y_next, z_next = ego_trajectory[i+1].coordinates
        timestamp_next = ego_trajectory[i+1].timestamp
        distance = compute_distance((x,y), (x_next, y_next))
        point_wise_speed.append(distance/(timestamp_next - timestamp).total_seconds())
    return point_wise_speed

def detection_to_img_segment(car_loc2d, cam_segment_mapping):
    for mapping in cam_segment_mapping:
        cam_segment, road_segment_info = mapping
        if Polygon(mapping).contains(Point(car_loc2d)):
            return mapping

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
    return database._execute_query(query)

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
        for i in range(len(car_trajectory)):
            if (car_trajectory[i].timestamp > current_time and
               not point_in_polygon(point.coordinates, segmentpolygon)):
                return point.timestamp
        return None
    segmentheading = current_segment_info.segment_heading
    car_vector = (car_loc.x + cos(math.radians(segmentheading)), car_loc.y + sin(math.radians(segmentheading)))
    car_heading_line = LineString([car_loc, car_vector])
    intersection = segmentpolygon.intersection(car_heading_line)
    assert len(intersection) == 0 or len(intersection) == 2
    if len(intersection) == 0:
        raise ValueError("The car is not drivable in the segment")
    elif len(intersection) == 2:
        distance1 = compute_distance(car_loc, intersection[0])
        distance2 = compute_distance(car_loc, intersection[1])
        if distance1 < distance2:
            return distance1 / max_car_speed(current_segment_info.road_type)
        else:
            return distance2 / max_car_speed(current_segment_info.road_type)


def meetup(car1_loc,
           car2_loc,
           car1_heading,
           car2_heading,
           road_type,
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
    if car1_trajectory is not None and car2_trajectory is None:
        car2_speed = max_car_speed(road_type) if car2_speed is None else car2_speed
        car2_vector = (car2_loc.x + cos(math.radians(car2_heading)), car2_loc.y + sin(math.radians(car2_heading)))
        car2_heading_line = LineString([car2_loc, car_vector])
        car1_trajectory = LineString(car1_trajectory)
        intersection = car1_trajectory.intersection(car2_heading_line)
        if len(intersection) == 1: # i.e. one car drives towards south, the other towards east
            meetup_point = intersection[0]
            time1 = point_to_nearest_trajectory_timestamp(meetup_point, car1_trajectory)
            distance2 = compute_distance(car2_loc, meetup_point)
            time2 = distance2 / car2_speed
            return (min(time1, time2), meetup_point)
        elif len(intersection) == 0: # i.e. one car drives towards south, the other towards north
            meetup_point = Point((car1_loc.x + car2_loc.x) / 2, (car1_loc.y + car2_loc.y) / 2)
            time1 = point_to_nearest_trajectory(meetup_point, car1_trajectory)
            distance2 = compute_distance(car2_loc, meetup_point)
            time2 = distance2 / car2_speed
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

def time_to_exit_view(ego_loc, car_loc, car_heading, ego_trajectory, view_distance):
    """Return the time, and location that the car goes beyond ego's view distance

    Assumption: car drives at max speed
    """
    ego_speed = get_ego_speed(ego_trajectory)
    car_speed = max_car_speed(road_type)
    exit_view_time = view_distance/car_speed - ego_speed
    return timestamp_to_nearest_trajectory(ego_trajectory, exit_view_time)

def relative_direction_to_ego(detection_info, ego_loc, ego_heading):
    """Return the relative direction to ego
       Now only support opposite and same direction
       TODO: add driving into and driving away from
    """
    relative_heading = abs(detection_info.heading - ego_heading) % 360
    if math.cos(math.radians(relative_heading)) > 0:
        return SAME_DIRECTION
    else:
        return OPPOSITE_DIRECTION

def compute_priority(all_detection_info):
    for detection_info in all_detection_info:
        detection_info.priority = detection_info.area/detection_info.distance
