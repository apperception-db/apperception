import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from pyquaternion import Quaternion

from ...camera_config import Float3, Float4, Float33
from ...payload import Payload
from ...types import DetectionId
from .detection_2d import Detection2D, Metadatum

signs = [-1, 1]
gp = []
for x in signs:
    for y in signs:
        for z in signs:
            gp.append([x, y, z])

get_points = np.array(gp)


def rotate(vectors: "npt.NDArray", rotation: "Quaternion") -> "npt.NDArray":
    """Rotate 3D Vector by rotation quaternion.
    Params:
        vectors: (3 x N) 3-vectors each specified as any ordered
            sequence of 3 real numbers corresponding to x, y, and z values.
        rotation: A rotation quaternion.

    Returns:
        The rotated vectors (3 x N).
    """
    return rotation.unit.rotation_matrix @ vectors


def conj(q: "npt.NDArray") -> "npt.NDArray":
    return np.concatenate([q[0:1, :], -q[1:, :]])


def _3d_to_2d(
    _translation: "Float3",
    _size: "Float3",
    _rotation: "Float4",
    _camera_translation: "Float3",
    _camera_rotation: "Quaternion",
    _camera_intrinsics: "Float33"
) -> "Float4":
    translation = np.array(_translation)
    size = np.array((_size[1], _size[0], _size[2])) / 2.
    rotation = Quaternion(_rotation)
    camera_translation = np.array(_camera_translation)
    camera_rotation = _camera_rotation
    camera_intrinsics = np.array(_camera_intrinsics)

    points = size * get_points

    translations = rotate(points.T, rotation).T + translation

    points_from_camera = rotate((translations - camera_translation).T, camera_rotation.inverse)

    pixels = camera_intrinsics @ points_from_camera
    pixels /= pixels[2:3]

    xs = pixels[0].tolist()
    ys = pixels[1].tolist()

    left = min(xs)
    top = min(ys)
    right = max(xs)
    bottom = max(ys)
    return left, top, right, bottom


classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
           'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
           'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
           'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
           'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
           'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
           'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
           'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
           'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
           'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']


CLASS_MAP = {
    'human.pedestrian.adult': 0,
    'human.pedestrian.child': 0,
    'human.pedestrian.construction_worker': 0,
    'human.pedestrian.personal_mobility': 0,
    'human.pedestrian.police_officer': 0,
    'vehicle.bicycle': 1,
    'vehicle.bus.bendy': 5,
    'vehicle.bus.rigid': 5,
    'vehicle.car': 2,
    'vehicle.motorcycle': 3,
    'vehicle.trailer': 7,
    'vehicle.truck': 7,
}


class GroundTruthDetection(Detection2D):
    def __init__(self, df_annotations: "pd.DataFrame"):
        annotations: "list[dict]" = df_annotations.to_dict('records')
        self.annotation_map: "dict[str, list[dict]]" = {}

        classes: "set[str]" = set()
        for a in annotations:
            fids: "list[str]" = a['sample_data_tokens']
            for fid in fids:
                if fid not in self.annotation_map:
                    self.annotation_map[fid] = []
                self.annotation_map[fid].append(a)

            classes.add(a['category'])

        self.id_to_classes = [*classes]
        self.class_to_id = {
            c: i
            for i, c
            in enumerate(self.id_to_classes)
        }

    def _run(self, payload: "Payload"):
        metadata: "list[Metadatum | None]" = []
        dimension = payload.video.dimension
        for i, cc in enumerate(payload.video._camera_configs):
            fid = cc.frame_id
            assert isinstance(fid, str)
            annotations = self.annotation_map.get(fid, [])
            tensor = []
            ids = []
            for a in annotations:
                if a['category'] not in CLASS_MAP:
                    continue

                left, top, right, bottom = _3d_to_2d(
                    a['translation'],
                    a['size'],
                    a['rotation'],
                    cc.camera_translation,
                    cc.camera_rotation,
                    cc.camera_intrinsic
                )

                left = max(0, min(left, dimension[0] - 1))
                right = max(0, min(right, dimension[0] - 1))
                top = max(0, min(top, dimension[1] - 1))
                bottom = max(0, min(bottom, dimension[1] - 1))

                d2d = [left, top, right, bottom]
                tensor.append([*d2d, 1, CLASS_MAP[a['category']]])
                ids.append(a['token'])

            if len(tensor) == 0:
                metadata.append(Metadatum(torch.Tensor([]), classes, []))
            else:
                metadata.append(Metadatum(torch.Tensor(tensor), classes, [DetectionId(i, _id) for _id in ids]))

        return None, {self.classname(): metadata}
