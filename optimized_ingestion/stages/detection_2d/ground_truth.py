from typing import Any
import torch
import numpy as np
import numpy.typing as npt
from pyquaternion import Quaternion

from ...camera_config import Float3, Float4, Float33

from ...payload import Payload
from .detection_2d import Detection2D


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
    _camera_rotation: "Float4",
    _camera_intrinsics: "Float33"
) -> "torch.Tensor":
    translation = np.array(_translation)
    size = np.array(_size)
    rotation = Quaternion(_rotation)
    camera_translation = np.array(_camera_translation)
    camera_rotation = Quaternion(_camera_rotation)
    camera_intrinsics = np.array(_camera_intrinsics)

    points = size * get_points

    translations = rotate(points.T, rotation).T + translation

    points_from_camera = rotate((translations - camera_translation).T, camera_rotation)

    pixels = camera_intrinsics @ points_from_camera
    pixels /= pixels[2:3]


    pass



class GroundTruthDetection(Detection2D):
    def __init__(self, annotations: "list[Any]"):
        self.annotations = annotations
        self.annotation_map = {}

        classes: "set[str]" = set()
        for a in annotations:
            fid = a['sample_data_tokens']
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
        for cc in payload.video._camera_configs:
            fid = cc.frame_id
            annotations = self.annotation_map[fid]
            for a in annotations:
                
        pass
