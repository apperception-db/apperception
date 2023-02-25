import torch

from ...payload import Payload
from .detection_2d import Detection2D, Metadatum


class ObjectTypeFilter(Detection2D):
    def __init__(self, types: "list[str]"):
        self.types = types

    def _run(self, payload: "Payload"):
        detection_2d = Detection2D.get(payload)
        assert detection_2d is not None

        assert len(detection_2d) != 0
        _, class_mapping = detection_2d[0]
        if isinstance(class_mapping, dict):
            class_mapping = list(class_mapping.values())
        assert isinstance(class_mapping, list)
        type_indices_to_keep: "set[int]" = set()

        for t in self.types:
            idx = class_mapping.index(t)
            type_indices_to_keep.add(idx)

        metadata = []
        for keep, (det, _) in zip(payload.keep, detection_2d):
            if not keep:
                metadata.append(Metadatum(torch.Tensor([]), class_mapping))
                continue

            type_indices = det[:, 5]
            det_to_keep: "list[int]" = []
            type_indices_list: "list[float]" = type_indices.tolist()
            for i, type_index in enumerate(type_indices_list):
                assert isinstance(type_index, float)
                assert type_index.is_integer()
                if type_index in type_indices_to_keep:
                    det_to_keep.append(i)

            metadata.append(Metadatum(det[det_to_keep], class_mapping))
        return None, {ObjectTypeFilter.classname(): metadata}
