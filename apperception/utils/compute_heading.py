from typing import List
import numpy as np


def compute_heading(trajectories):
    headings = []
    for traj in trajectories:
        traj = traj[0]
        heading: List[float] = []
        for j in range(1, len(traj)):
            prev_pos = traj[j - 1]
            current_pos = traj[j]
            heading.append(0)
            if current_pos[1] != prev_pos[1]:
                heading[j] = np.arctan2(current_pos[1] - prev_pos[1], current_pos[0] - prev_pos[0])
            heading[j] *= 180 / np.pi  # convert to degrees from radian
            heading[j] = (heading[j] + 360) % 360  # converting such that all headings are positive
        headings.append(heading)
    return headings
