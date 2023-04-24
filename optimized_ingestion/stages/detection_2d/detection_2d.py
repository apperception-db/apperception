import torch
from typing import NamedTuple

from ...types import DetectionId
from ..stage import Stage


class Metadatum(NamedTuple):
    """
    detections:
    -----------
    A torch.Tensor with size N x 6, representing N objects detected from a single frame.

    Each column represents:
    - bbox_left
    - bbox_top
    - bbox_right
    - bbox_bottom
    - conf
    - class

    class_map:
    ----------
    A mapping from an object class number (in detections[:, 5]) to a class string name
    """
    detections: "torch.Tensor"
    class_map: "list[str]"
    detection_ids: "list[DetectionId]"


class Detection2D(Stage[Metadatum]):
    pass
