from typing import TYPE_CHECKING, List, Union

if TYPE_CHECKING:
    from psycopg2 import connection as Connection

    from apperception.data_types import FetchCameraTuple


def fetch_camera(
    conn: "Connection", scene_name: str, frame_timestamps: Union[List[str], List[int]]
) -> List["FetchCameraTuple"]:
    """
    TODO: Fix fetch camera that given a scene_name and frame_num, return the corresponding camera metadata
    scene_name: str
    frame_num: int[]
    return a list of metadata info for each frame_num
    """

    cursor = conn.cursor()
    # query = '''SELECT camera_info from camera_table where camera_table.camera_id == scene_name and camera_table.frame_num in frame_num'''
    # if cam_id == []:
    # 	query = '''SELECT cameraId, ratio, ST_X(origin), ST_Y(origin), ST_Z(origin), ST_X(focalpoints), ST_Y(focalpoints), fov, skev_factor ''' \
    # 	 + '''FROM Cameras WHERE worldId = \'%s\';''' %world_id
    # else:
    # 	query = '''SELECT cameraId, ratio, ST_X(origin), ST_Y(origin), ST_Z(origin), ST_X(focalpoints), ST_Y(focalpoints), fov, skev_factor ''' \
    # 	 + '''FROM Cameras WHERE cameraId IN (\'%s\') AND worldId = \'%s\';''' %(','.join(cam_id), world_id)
    # TODO: define ST_XYZ somewhere else
    query = f"""
    CREATE OR REPLACE FUNCTION ST_XYZ (g geometry) RETURNS real[] AS $$
        BEGIN
            RETURN ARRAY[ST_X(g), ST_Y(g), ST_Z(g)];
        END;
    $$ LANGUAGE plpgsql;

    SELECT
        cameraId,
        ST_XYZ(egoTranslation),
        egoRotation,
        ST_XYZ(cameraTranslation),
        cameraRotation,
        cameraIntrinsic,
        frameNum,
        fileName,
        cameraHeading,
        egoHeading
    FROM Cameras
    WHERE
        cameraId = '{scene_name}' AND
        timestamp IN ({",".join(map(str, frame_timestamps))})
    ORDER BY cameraId ASC, frameNum ASC;
    """
    # print(query)
    cursor.execute(query)
    result: list = cursor.fetchall()
    return result
