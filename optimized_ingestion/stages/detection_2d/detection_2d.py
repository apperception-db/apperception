import torch
from typing import List, Tuple

from ..stage import Stage

# tensor dimension n x 6:
#   - bbox_left
#   - bbox_top
#   - bbox_w
#   - bbox_h
#   - conf
#   - class
Metadatum = Tuple[torch.Tensor, List[str]]


class Detection2D(Stage[Metadatum]):
    pass
