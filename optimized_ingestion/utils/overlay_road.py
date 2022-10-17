import cv2
import numpy as np
from pyquaternion import Quaternion
from apperception.database import database
from .iterate_video import iterate_video
from ..payload import Payload


def overlay_road(payload: "Payload"):
    video = payload.video

    database._execute_query("""
    DROP FUNCTION IF EXISTS viewAngleLine(real, geometry, real, real);
    CREATE OR REPLACE FUNCTION viewAngleLine(
        view_point_heading real,
        view_point geometry,
        max_distance real,
        view_angle real
    ) RETURNS geometry AS
    $BODY$
    BEGIN
    RETURN ST_Translate(
        ST_Rotate(
            ST_MakeLine(
                ST_MakePoint(0, 0),
                ST_MakePoint(0, max_distance)
            ),
            radians(view_point_heading + view_angle)
        ),
        ST_X(view_point),
        ST_Y(view_point)
    );
    END
    $BODY$
    LANGUAGE 'plpgsql';
    """)

    cap = cv2.VideoCapture(video.videofile)
    for frame, img in zip(video, iterate_video(cap)):
        translation = frame.camera_translation
        translation_str = f"'Point Z ({' '.join(map(str, translation))})'"
        heading = frame.camera_heading
        polygons = database._execute_query(f"""
        SELECT elementPolygon
        FROM SegmentPolygon
        WHERE 'intersection' = Any(segmentTypes)
            AND ST_Distance({translation_str}, elementPolygon) < 100
            AND (
                viewAngle(ST_Centroid(elementPolygon), {heading}, {translation_str}) < 35
                OR ST_Intersects(elementPolygon, viewAngleLine({heading}, {translation_str}, 100, -35))
                OR ST_Intersects(elementPolygon, viewAngleLine({heading}, {translation_str}, 100, -35))
            )
        """)

        height, width = img.shape[:2]
        intrinsic = frame.camera_intrinsic
        [[fx, _, x0], [_, fy, y0], [_, _, s]] = intrinsic
        rotation = Quaternion(frame.camera_rotation)

        ys = []
        xs = []
        Zs = []
        Xs = []
        Ys = []
        for y in range(height):
            for x in range(width):
                direction = np.array([(s * x + 0.5 - x0) / fx, (s * y + 0.5 - y0) / fy, 1])
                direction = rotation.rotate(direction)

                t = -translation[2] / direction[2]
                Z = 0
                X = direction[0] * t + translation[0]
                Y = direction[1] * t + translation[1]
                ys.append(y)
                xs.append(x)
                Zs.append(Z)
                Xs.append(X)
                Ys.append(Y)

                # for (polygon,) in polygons:
                #     coords: "Tuple[Tuple[float, float], ...]" = polygon.coords
                #     return polygon
        database._execute_query(f"""
        SELECT elementPolygon
        FROM SegmentPolygon
        JOIN UNNEST(
            ARRAY[{",".join(map(str, xs))}],
            ARRAY[{",".join(map(str, ys))}],
            ARRAY[{",".join(map(str, Xs))}],
            ARRAY[{",".join(map(str, Ys))}],
            ARRAY[{",".join(map(str, Zs))}]
        ) AS coords(px, py, x, y, z)
        ON ....
        WHERE 'intersection' = Any(segmentTypes)
            AND ST_Distance({translation_str}, elementPolygon) < 100
            AND (
                viewAngle(ST_Centroid(elementPolygon), {heading}, {translation_str}) < 35
                OR ST_Intersects(elementPolygon, viewAngleLine({heading}, {translation_str}, 100, -35))
                OR ST_Intersects(elementPolygon, viewAngleLine({heading}, {translation_str}, 100, -35))
            )
        """)
