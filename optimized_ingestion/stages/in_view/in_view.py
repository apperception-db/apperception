import numpy as np
import numpy.typing as npt
from bitarray import bitarray
from postgis import MultiPoint
from psycopg2 import sql
from pyquaternion import Quaternion

from apperception.database import database

from ...payload import Payload
from ..stage import Stage


class InView(Stage):
    def __init__(self, distance: float, segment_type: "str | list[str]"):
        super().__init__()
        self.distance = distance
        self.segment_types = segment_type if isinstance(segment_type, list) else [segment_type]

    def _run(self, payload: "Payload") -> "tuple[bitarray, None]":
        w, h = payload.video.dimension
        Z = self.distance
        view_vertices_2d = np.array([
            # 4 corners of the image frame
            (w, h, 1),
            (w, 0, 1),
            (0, h, 1),
            (0, 0, 1),
            # camera position
            (0, 0, 0),
        ]).T
        assert view_vertices_2d.shape == (3, 5), view_vertices_2d.shape

        [[fx, s, x0], [_, fy, y0], [_, _, _]] = payload.video.interpolated_frames[0].camera_intrinsic

        # 3x3 matrix to convert points from pixel-coordinate to camera-coordinate
        pixel2camera = Z * np.array([
            [1 / fx, -s / (fx * fy), (s * y0 / (fx * fy)) - (x0 / fx)],
            [0, 1 / fy, -y0 / fy],
            [0, 0, 1]
        ])
        assert pixel2camera.shape == (3, 3), pixel2camera.shape

        view_vertices_from_camera = pixel2camera @ view_vertices_2d
        assert view_vertices_from_camera.shape == (3, 5), view_vertices_from_camera.shape

        extrinsics: "list[npt.NDArray]" = []
        indices: "list[int]" = []
        for i, (k, f) in enumerate(zip(payload.keep, payload.video.interpolated_frames)):
            if not k:
                continue

            rotation = Quaternion(f.camera_rotation)
            rotation_matrix = rotation.unit.rotation_matrix
            assert rotation_matrix.shape == (3, 3), rotation_matrix.shape

            # 3x4 matrix to convert points from camera-coordinate to world-coordinate
            translation = np.array(f.camera_translation)[np.newaxis].T
            extrinsic = np.hstack((rotation_matrix, translation))
            assert extrinsic.shape == (3, 4), extrinsic.shape

            extrinsics.append(extrinsic)
            indices.append(i)

        N = len(extrinsics)

        # add 1 to the last row
        view_vertices_from_camera = np.concatenate((
            view_vertices_from_camera,
            np.ones_like(view_vertices_from_camera[:1]),
        ))

        _extrinsics = np.stack(extrinsics)
        assert _extrinsics.shape == (N, 3, 4), _extrinsics.shape

        # convert 4 corner points from pixel-coordinate to world-coordinate
        view_area_3ds = _extrinsics @ view_vertices_from_camera
        assert view_area_3ds.shape == (N, 3, 5), view_area_3ds.shape

        # project view_area to 2D from top-down view
        view_area_2ds = view_area_3ds[:, :2].swapaxes(1, 2)
        assert view_area_2ds.shape == (N, 5, 2), view_area_2ds.shape

        assert any(
            np.array_equal(view_area_3ds[n, :2, i], view_area_2ds[n, i])
            for n in range(N)
            for i in range(5)
        ), (view_area_3ds, view_area_2ds)

        view_areas: "list[MultiPoint]" = []
        for i, view_area_2d in zip(indices, view_area_2ds):
            view_area = MultiPoint(view_area_2d.tolist())
            view_areas.append(view_area)

        # TODO: where clause should depends on query predicate
        results = database.execute(sql.SQL("""
        SELECT index
        FROM UNNEST (
            {view_areas},
            {indices}::int[]
        ) AS ViewArea(points, index)
        JOIN SegmentPolygon ON ST_Intersects(ST_ConvexHull(points), elementPolygon)
        WHERE {segment_type}
        """).format(
            view_areas=sql.Literal(view_areas),
            indices=sql.Literal(indices),
            segment_type=sql.SQL(" OR ".join(map(roadtype, self.segment_types)))
        ))

        keep = bitarray(len(payload.keep))
        keep.setall(0)
        for (index, ) in results:
            keep[index] = 1

        return keep, None


def roadtype(t: "str"):
    return f"__roadtype__{t}__"
