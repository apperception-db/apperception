from typing import TYPE_CHECKING
from apperception.database import database
import numpy as np
from pyquaternion import Quaternion
import cv2
from tqdm import tqdm

from ..utils.iterate_video import iterate_video


from ..stages.tracking_3d.from_2d_and_road import rotate

if TYPE_CHECKING:
    from ..payload import Payload


def overlay_roads(payload: "Payload", filename: str) -> None:
    video = cv2.VideoCapture(payload.video.videofile)

    images = []
    for frame, img in tqdm(zip(payload.video, iterate_video(video)), total=len(payload.video)):
        camera_translation = to_point(frame.camera_translation)
        [[fx, _, x0], [_, fy, y0], [_, _, s]] = frame.camera_intrinsic
        points = []

        # TODO: use matrix
        for y in range(900):
            for x in range(1600):
                points.append((
                    (s * x - x0) / fx,
                    (s * y - y0) / fy,
                    1
                ))

        np_points = np.array(points).T
        rotated_directions = rotate(np_points, Quaternion(frame.camera_rotation).unit)
        ts = -frame.camera_translation[2] / rotated_directions[2, :]
        _points = (rotated_directions * ts + np.array(frame.camera_translation)[:, np.newaxis]).T
        points_str = [*map(to_point, _points)]
        XYs = database._execute_query(f"""
            SELECT ST_X(p), ST_Y(p)
            FROM UNNEST(
                ARRAY[{",".join(points_str)}]
            ) as points(p)
            WHERE EXISTS (
                SELECT TRUE
                FROM SegmentPolygon
                WHERE
                    ST_Distance({camera_translation}, elementPolygon) < 100
                AND
                    ST_Covers(elementPolygon, p)
            )
        """)

        XYs_np = np.array(XYs).T
        img[XYs_np[0], XYs_np[1], 2] = 255
        images.append(img)

    height, width, _ = images[0].shape
    out = cv2.VideoWriter(
        filename, cv2.VideoWriter_fourcc(*"mp4v"), int(payload.video.fps), (width, height)
    )
    for image in tqdm(images):
        out.write(image)
    out.release()
    cv2.destroyAllWindows()


def to_point(point):
    x, y, z = point
    return f"st_pointz({x}, {y}, {z})"
