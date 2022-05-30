import numpy as np


def create_transform_matrix(
    focal_x: float, focal_y: float, cam_x: float, cam_y: float, skew_factor: float
) -> np.ndarray:
    return np.array([[focal_x, skew_factor, cam_x, 0], [0, focal_y, cam_y, 0], [0, 0, 1, 0]])
