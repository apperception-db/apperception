from typing import List


def bbox_to_data3d(bbox: List[List[float]]):
    """
    Compute the center, x, y, z delta of the bbox
    """
    tl, br = bbox
    x_delta = (br[0] - tl[0]) / 2
    y_delta = (br[1] - tl[1]) / 2
    z_delta = (br[2] - tl[2]) / 2
    center = (tl[0] + x_delta, tl[1] + y_delta, tl[2] + z_delta)

    return center, x_delta, y_delta, z_delta
