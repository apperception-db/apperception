import torch
from typing import List, Tuple

from ..stage import Stage

# tensor dimension n x 18:
#   - bbox_left
#   - bbox_top
#   - bbox_w
#   - bbox_h
#   - conf
#   - class
#   - bbox3d_left_x
#   - bbox3d_left_y
#   - bbox3d_left_z
#   - bbox3d_right_x
#   - bbox3d_right_y
#   - bbox3d_right_z
#   - bbox3d_from_camera_left_x
#   - bbox3d_from_camera_left_y
#   - bbox3d_from_camera_left_z
#   - bbox3d_from_camera_right_x
#   - bbox3d_from_camera_right_y
#   - bbox3d_from_camera_right_z
Metadatum = Tuple[torch.Tensor, List[str]]


class Detection3D(Stage[Metadatum]):
    pass
