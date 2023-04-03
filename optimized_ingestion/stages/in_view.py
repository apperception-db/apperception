import numpy as np
import numpy.typing as npt
from bitarray import bitarray
from pyquaternion import Quaternion
from postgis import MultiPoint
from psycopg2 import sql

from apperception.database import database

from .stage import Stage
from ..payload import Payload


class InView(Stage):
    def __init__(self, distance: float, segment_type: "str | list[str]"):
        super().__init__()
        self.distance = distance
        self.segment_type = segment_type if isinstance(segment_type, list) else [segment_type]
    
    def _run(self, payload: "Payload") -> "tuple[bitarray, None]":
        width, height = payload.video.dimension
        Z = self.distance
        point_2ds = Z * np.array([
            # 4 corners of the image frame
            (width, height, 1),
            (width, 0, 1),
            (0, height, 1),
            (0, 0, 1),
            # camera position
            (0, 0, 0),
        ]).T
        assert point_2ds.shape == (3, 5), point_2ds.shape
        
        pixel2worlds: "list[npt.NDArray]" = []
        indices: "list[int]" = []
        for i, (k, f) in enumerate(zip(payload.keep, payload.video.interpolated_frames)):
            if not k: continue

            rotation = Quaternion(f.camera_rotation)
            rotation_matrix = rotation.unit.rotation_matrix
            assert rotation_matrix.shape == (3, 3), rotation_matrix.shape

            [[fx, _, x0], [_, fy, y0], [_, _, s]] = f.camera_intrinsic
            # 3x3 matrix to convert points from pixel-coordinate to camera-coordinate
            pixel2camera = np.array([
                [s / fx, 0, -x0 / fx],
                [0, s / fy, -y0 / fy],
                [0, 0, 1]
            ])
            assert pixel2camera.shape == (3, 3), pixel2camera.shape

            # 3x3 matrix to convert points from pixel-coordinate to world-coordinate from the camera position
            pixel2world = rotation_matrix @ pixel2camera
            assert pixel2world.shape == (3, 3), pixel2world.shape

            # 3x4 matrix to convert points from pixel-coordinate to world-coordinate
            translation = np.array(f.camera_translation)[np.newaxis].T
            pixel2world = np.hstack((pixel2world, translation))
            assert pixel2world.shape == (3, 4), pixel2world.shape

            pixel2worlds.append(pixel2world)
            indices.append(i)
        
        N = len(pixel2worlds)
        
        # add 1 to the last row
        _point_2ds = np.concatenate((point_2ds, np.ones_like(point_2ds[:1])))
        assert _point_2ds.shape == (4, 5), _point_2ds.shape

        _pixel2worlds = np.stack(pixel2worlds)
        assert _pixel2worlds.shape == (N, 3, 4), _pixel2worlds.shape

        # convert 4 corner points from pixel-coordinate to world-coordinate
        view_area_3ds = _pixel2worlds @ _point_2ds
        assert view_area_3ds.shape == (N, 3, 5), view_area_3ds.shape

        # project view_area to 2D from top-down view
        view_area_2ds = view_area_3ds[:, :2].reshape((-1, 5, 2))
        assert view_area_2ds.shape == (N, 5, 2), view_area_2ds.shape

        assert any(
            np.array_equal(view_area_3ds[n, :2, i], view_area_2ds[n, i])
            for n in range(N)
            for i in range(5)
        )

        view_areas: "list[MultiPoint]" =  []
        for i, view_area_2d in zip(indices, view_area_2ds):
            view_area = MultiPoint(view_area_2d.tolist())
            view_areas.append(view_area)
        
        # TODO: where clause should depends on query predicate
        results = database.execute(sql.SQL("""
        SELECT index
        FROM UNNEST (
            {view_areas}::MultiPoint[],
            {indices}::int[]
        ) as ViewArea(points, index)
        JOIN SegmentPolygon
        WHERE segmentTypes && {segment_type}
        AND ST_Intersects(ST_ConvexHull(points), elementPolygon)
        """).format(
            view_areas=sql.Literal(view_areas),
            indices=sql.Literal(indices),
            segment_type=sql.Literal(self.segment_type)
        ))

        keep = bitarray(len(payload.keep))
        keep.setall(0)
        for (index, ) in results:
            keep[index] = 1

        return keep, None
