import torch
from typing import List, Tuple

from ..stage import Stage

Metadatum = Tuple[torch.Tensor, List[str]]


class Detection3D(Stage[Metadatum]):
    pass
