import numpy as np


def world_to_pixel(world_coords: np.ndarray, transform: np.ndarray) -> np.ndarray:
    tl_x, tl_y, tl_z, br_x, br_y, br_z = world_coords.T

    tl_world_pixels = np.array([tl_x, tl_y, tl_z, np.ones(len(tl_x))])
    tl_vid_coords = transform @ tl_world_pixels

    br_world_pixels = np.array([br_x, br_y, br_z, np.ones(len(br_x))])
    br_vid_coords = transform @ br_world_pixels

    return np.stack(
        (tl_vid_coords[0], tl_vid_coords[1], br_vid_coords[0], br_vid_coords[1]), axis=0
    )
