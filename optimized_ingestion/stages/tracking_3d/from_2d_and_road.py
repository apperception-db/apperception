from typing import TYPE_CHECKING, Dict, List

import numpy as np
import numpy.typing as npt
from pyquaternion import Quaternion
from tqdm import tqdm

from ...payload import Payload
from ..tracking_2d import Tracking2D
from ..utils.is_annotated import is_annotated
from .tracking_3d import Tracking3D, Tracking3DResult

if TYPE_CHECKING:
    from ...trackers.yolov5_strongsort_osnet_tracker import TrackingResult
    from ..stage import StageOutput


class From2DAndRoad(Tracking3D):
    def __call__(self, payload: "Payload") -> "StageOutput":
        if not is_annotated(Tracking2D, payload):
            # payload = payload.filter(Tracking2D())
            raise Exception()

        metadata: "List[Dict[float, Tracking3DResult] | None]" = []
        trajectories: "Dict[float, List[Tracking3DResult]]" = {}
        video = payload.video
        trackings: "List[Dict[float, TrackingResult] | None]" = Tracking2D.get(payload.metadata)
        for i, (k, tracking, frame) in tqdm(enumerate(zip(payload.keep, trackings, video)), total=len(payload.keep)):
            if not k or tracking is None:
                metadata.append(None)
                continue

            if len(tracking) == 0:
                metadata.append({})
                continue

            trackings3d: "Dict[float, Tracking3DResult]" = {}
            [[fx, _, x0], [_, fy, y0], [_, _, s]] = frame.camera_intrinsic
            rotation = Quaternion(frame.camera_rotation).unit
            translation = np.array(frame.camera_translation)

            ids: "List[float]" = []
            dirx = []
            diry = []
            N = len(tracking)
            for oid, t in tracking.items():
                ids.append(oid)
                dirx.append(t.bbox_left + t.bbox_w / 2)
                diry.append(t.bbox_top + t.bbox_h)

            directions = np.stack([
                (s * np.array(dirx) - x0) / fx,
                (s * np.array(diry) - y0) / fy,
                np.ones(N),
            ])
            rotated_directions = rotate(directions, rotation)
            ts = -translation[2] / rotated_directions[2, :]

            XY = rotated_directions[:2, :] * ts + translation[:2, np.newaxis]
            points = np.concatenate([XY, np.zeros((1, N))])
            points_from_camera = rotate(points - translation.T[:, np.newaxis], rotation.inverse)

            for oid, point, point_from_camera in zip(ids, points.T, points_from_camera.T):
                assert point_from_camera.shape == (3,)
                point_from_camera = (point_from_camera[0], point_from_camera[1], point_from_camera[2])
                trackings3d[oid] = Tracking3DResult(oid, point_from_camera, point)
                if oid not in trajectories:
                    trajectories[oid] = []
                trajectories[oid].append(trackings3d[oid])
            metadata.append(trackings3d)

        for trajectory in trajectories.values():
            last = len(trajectory) - 1
            for i, t in enumerate(trajectory):
                if i > 0:
                    t.prev = trajectory[i - 1]
                if i < last:
                    t.next = trajectory[i + 1]

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
    N = vectors.shape[1]

    # Get rotation matrix
    # 4 x 4
    rmatrix = rotation.unit._q_matrix()

    # Create quaternions from vectors
    # 4 x N
    qvectors = np.concatenate([np.zeros((1, N)), vectors])

    # rotated vectors is Q * v * Q^{-1} -----> (Q * (Q * v)^{-1})^{-1}
    # 4 x N
    ret = conj(rmatrix @ conj(rmatrix @ qvectors))

    # assert np.allclose(np.zeros((N,)), a[0, :])
    # assert np.allclose(ret[1:, :], slow_rotate_multiple(vectors, rotation))
    return ret[1:, :]


def slow_rotate(vectors: "npt.NDArray", rotation: "Quaternion") -> "npt.NDArray":
    out = []
    for v in vectors.T:
        out.append(rotation.rotate(v))
    return np.array(out).T


def conj(q: "npt.NDArray") -> "npt.NDArray":
    return np.concatenate([q[0:1, :], -q[1:, :]])
