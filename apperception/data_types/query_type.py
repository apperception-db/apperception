from enum import IntEnum


class QueryType(IntEnum):
    # query type: for example, if we call get_cam(), and we execute the commands from root. when we encounter
    # recognize(), we should not execute it because the inserted object must not be in the final result. we use enum
    # type to determine whether we should execute this node
    CAM, BBOX, TRAJ = 0, 1, 2
