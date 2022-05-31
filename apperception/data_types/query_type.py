from typing import Literal, Union


QueryType = Union[
    Literal["CAM"],
    Literal["BBOX"],
    Literal["TRAJ"],
    Literal["METADATA"],
]
