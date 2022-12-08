import torch
from typing import List, Tuple

from ..stage import Stage

Metadatum = Tuple[torch.Tensor, List[str]]


class Detection2D(Stage[Metadatum]):
    pass
