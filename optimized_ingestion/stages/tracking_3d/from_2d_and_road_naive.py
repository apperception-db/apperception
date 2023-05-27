import numpy as np
import numpy.typing as npt
from pyquaternion import Quaternion
from tqdm import tqdm
from typing import TYPE_CHECKING, Dict, List

from ...payload import Payload
from ..tracking_2d.tracking_2d import Tracking2D
from ..utils.is_annotated import is_annotated
from .tracking_3d import Tracking3D, Tracking3DResult

if TYPE_CHECKING:
    from ...trackers.yolov5_strongsort_osnet_tracker import TrackingResult
    from ..stage import StageOutput


class From2DAndRoadNaive(Tracking3D):
    def _run(self, payload: "Payload") -> "StageOutput":
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
            rotation = frame.camera_rotation
            translation = np.array(frame.camera_translation)

            for oid, t in tracking.items():
                x = t.bbox_left + t.bbox_w / 2
                y = t.bbox_top + t.bbox_h

                x = (s * x - x0) / fx
                y = (s * y - y0) / fy
                z = 1

                direction = np.array([x, y, z])
                assert direction.shape == (3,)
                rotated_direction = rotation.rotate(direction)
                assert rotated_direction.shape == (3,)

                # find t that z=0
                t = -translation[2] / rotated_direction[2]

                point = rotated_direction * t + translation
                assert point.shape == (3,)
                point_from_camera = direction * t
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
    _3, N = vectors.shape
    assert _3 == 3

    # Get rotation matrix
    # 4 x 4
    rmatrix = rotation.unit._q_matrix()
    assert rmatrix.shape == (4, 4)

    # Create quaternions from vectors
    # 4 x N
    qvectors = np.concatenate([np.zeros((1, N)), vectors])
    assert qvectors.shape == (4, N)

    # rotated vectors is Q * v * Q^{-1} -----> (Q * (Q * v)^{-1})^{-1}
    # 4 x N
    ret = conj(rmatrix @ conj(rmatrix @ qvectors))
    assert ret.shape == (4, N)

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