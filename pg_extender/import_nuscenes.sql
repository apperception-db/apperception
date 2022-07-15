DROP FUNCTION IF EXISTS import_nuscenes(text);
CREATE OR REPLACE FUNCTION import_nuscenes(file_directory text) RETURNS void AS
$BODY$
BEGIN
    EXECUTE 'CREATE TABLE IF NOT EXISTS Test_Scenic_Item_General_Trajectory(
                itemId TEXT,
                objectType TEXT,
                frameId TEXT,
                color TEXT,
                trajCentroids tgeompoint,
                largestBbox stbox,
                PRIMARY KEY (itemId)
                );';

    EXECUTE 'CREATE TABLE IF NOT EXISTS Test_Scenic_General_Bbox(
                itemId TEXT,
                trajBbox stbox,
                FOREIGN KEY(itemId)
                    REFERENCES Test_Scenic_Item_General_Trajectory(itemId)
                );';

    EXECUTE 'CREATE TABLE IF NOT EXISTS Test_Scenic_Cameras(
                cameraId TEXT,
                worldId TEXT,
                frameId TEXT,
                frameNum Int,
                fileName TEXT,
                cameraTranslation geometry,
                cameraRotation geometry,
                cameraIntrinsic real[][],
                egoTranslation geometry,
                egoRotation geometry,
                timestamp TEXT
                );';
    EXECUTE format('copy from stdin test_scenic_item_general_trajectory from %L with delimiter '','' quote ''"'' csv ', $1 || '/trajectory.csv');
    EXECUTE format('copy from stdin test_scenic_general_bbox from %L with delimiter '','' quote ''"'' csv ', $1 || '/bbox.csv');
    EXECUTE format('copy from stdin test_scenic_cameras from %L with delimiter '','' quote ''"'' csv ', $1 || '/camera.csv');
END;
$BODY$
LANGUAGE 'plpgsql';

