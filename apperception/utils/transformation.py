from typing import Any, Dict, Tuple

import numpy as np
import numpy.typing as npt
from pyquaternion import Quaternion


def transformation(
    copy_centroid_3d: "npt.NDArray[np.floating] | Tuple[float, float, float]",
    camera_config: Dict[str, Any],
) -> "npt.NDArray[np.floating]":
    """
    Transformation from 3d world coordinate to 2d frame coordinate given the camera config
    """
    centroid_3d: npt.NDArray[np.floating] = np.copy(copy_centroid_3d)

    centroid_3d -= camera_config["egoTranslation"]
    centroid_3d = np.dot(
        Quaternion(camera_config["egoRotation"]).inverse.rotation_matrix, centroid_3d
    )

    centroid_3d -= camera_config["cameraTranslation"]
    centroid_3d = np.dot(
        Quaternion(camera_config["cameraRotation"]).inverse.rotation_matrix, centroid_3d
    )

    view = np.array(camera_config["cameraIntrinsic"])
    viewpad = np.eye(4)
    viewpad[: view.shape[0], : view.shape[1]] = view

    # Do operation in homogenous coordinates.
    centroid_3d = centroid_3d.reshape((3, 1))
    centroid_3d = np.concatenate((centroid_3d, np.ones((1, 1))))
    centroid_3d = np.dot(viewpad, centroid_3d)
    centroid_3d = centroid_3d[:3, :]

    centroid_3d = centroid_3d / centroid_3d[2:3, :].repeat(3, 0).reshape(3, 1)
    return centroid_3d[:2, :]
