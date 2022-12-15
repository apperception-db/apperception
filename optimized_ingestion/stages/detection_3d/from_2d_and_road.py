import numpy as np
import numpy.typing as npt
import torch
from pyquaternion import Quaternion
from tqdm import tqdm
from typing import TYPE_CHECKING, Tuple

from ..detection_2d.detection_2d import Detection2D
from . import Detection3D

if TYPE_CHECKING:
    from ...payload import Payload


TO_BOTTOM_LEFT = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 1],
])
TO_BOTTOM_RIGHT = np.array([
    [1, 0, 1, 0],
    [0, 1, 0, 1],
])
TO_BOTTOM_CENTER = np.array([
    [1, 0, 0.5, 0],
    [0, 1, 0, 1],
])


class From2DAndRoad(Detection3D):
    def _run(self, payload: "Payload"):
        with torch.no_grad():
            detection2ds = Detection2D.get(payload)
            assert detection2ds is not None

            metadata: "list[Tuple[torch.Tensor, list[str]]]" = []
            for k, (d2d, clss), frame in tqdm(zip(payload.keep, detection2ds, payload.video)):
                if not k:
                    metadata.append((torch.tensor([], device=d2d.device), clss))
                
                device = d2d.device

                [[fx, _, x0], [_, fy, y0], [_, _, s]] = frame.camera_intrinsic
                rotation = frame.camera_rotation
                translation = np.array(frame.camera_translation)

                _, d = d2d.shape

                d2dt = d2d.T[:4, :].cpu().numpy()
                assert isinstance(d2dt, np.ndarray)

                _d, N = d2dt.shape
                assert _d == 4

                # TODO: should it be a 2D bbox in 3D?
                bottoms = np.concatenate([
                    TO_BOTTOM_LEFT @ d2dt,
                    TO_BOTTOM_RIGHT @ d2dt,
                ], axis=1)
                assert ((2, N * 2) == bottoms.shape), ((2, N * 2), bottoms.shape)
            
                directions = np.stack((
                    (s * bottoms[0] - x0) / fx,
                    (s * bottoms[1] - y0) / fy,
                    np.ones(N * 2),
                ))
                assert ((3, N * 2) == directions.shape), ((3, N * 2), directions.shape)

                rotated_directions = rotate(directions, rotation)

                # find t that z=0
                ts = -translation[2] / rotated_directions[2, :]

                points = rotated_directions * ts + translation[:, np.newaxis]
                points_from_camera = rotate(points - translation[:, np.newaxis], rotation.inverse)

                bbox3d = np.concatenate((
                    points[:, :N],
                    points[:, N:],
                ), axis=0).T
                assert ((N, 6) == bbox3d.shape), bbox3d.shape

                bbox3d_from_camera = np.concatenate((
                    points_from_camera[:, :N],
                    points_from_camera[:, N:],
                ), axis=0).T
                assert ((N, 6) == bbox3d_from_camera.shape), bbox3d_from_camera.shape

                d3d = torch.concatenate((
                    d2d,
                    torch.tensor(bbox3d, device=device),
                    torch.tensor(bbox3d_from_camera, device=device),
                ), dim=1)
                assert ((N, (d + 12)) == d3d.shape), d3d.shape

                metadata.append((d3d, clss))

            return None, {self.classname(): metadata}


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
