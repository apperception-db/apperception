DROP FUNCTION IF EXISTS materializeExistingTrajectory(text, text);
CREATE OR REPLACE FUNCTION materializeExistingTrajectory(itemId text, traj_to_materialize tgeompoint) RETURNS void AS
$BODY$
DECLARE
    interval_to_merge periodset;
    traj_record record;
BEGIN
    FOR traj_record IN EXECUTE E'SELECT Materialized_Trajectory.trajCentroids 
                                    FROM Materialized_Trajectory
                                    WHERE Materialized_Trajectory.itemId = \'' || $1 || E'\';'
    LOOP
        interval_to_merge := getTime(traj_record.trajCentroids) - getTime(traj_to_materialize) * getTime(traj_record.trajCentroids);
        traj_to_materialize := merge(traj_to_materialize, atPeriodSet(traj_record.trajCentroids, interval_to_merge));
    END LOOP;
    
    EXECUTE E'UPDATE Materialized_Trajectory
                SET trajCentroids = \'' || traj_to_materialize || E'\'
                WHERE itemId = \'' || $1 || E'\';';

END;
$BODY$
LANGUAGE 'plpgsql';

DROP FUNCTION IF EXISTS materializeNewTrajectory(text, tgeompoint);
CREATE OR REPLACE FUNCTION materializeNewTrajectory(itemId text, traj_to_materialize tgeompoint) RETURNS void AS
$BODY$
BEGIN
    EXECUTE E'INSERT INTO Materialized_Trajectory VALUES (\'' || $1 || E'\', \'' || traj_to_materialize || E'\');';

END;
$BODY$
LANGUAGE 'plpgsql';

DROP FUNCTION IF EXISTS materializeExistingBbox(text, stbox[]);
CREATE OR REPLACE FUNCTION materializeExistingBbox(itemId text, bbox_to_materialize stbox[]) RETURNS void AS
$BODY$
DECLARE
    materialized_bbox stbox [];
BEGIN
    materialized_bbox := ARRAY(SELECT Materialized_Bbox.trajBbox 
        FROM Materialized_Bbox 
        WHERE Materialized_Bbox.itemId = $1);
    materialized_bbox := merge(materialized_bbox, bbox_to_materialize);
    EXECUTE E'UPDATE Materialized_Bbox
                SET trajBbox = \'' || materialized_bbox || E'\'
                WHERE itemId = \'' || $1 || E'\';';
END;
$BODY$
LANGUAGE 'plpgsql';

DROP FUNCTION IF EXISTS materializeNewBbox(boolean);
CREATE OR REPLACE FUNCTION materializeNewBbox(reconcile_on_id boolean) RETURNS void AS
$BODY$
BEGIN
    IF reconcile_on_id THEN
        EXECUTE 'INSERT INTO Materialized_Bbox (itemId, trajBbox)
                    SELECT filtered_bbox.newId, ARRAY(SELECT Temp_Bbox.trajBbox 
                                                    FROM Temp_Bbox 
                                                    WHERE Temp_Bbox.cameraId = filtered_bbox.cameraId 
                                                    AND Temp_Bbox.itemId = filtered_bbox.itemId) as new_bboxes
                    FROM (SELECT distinct itemId as newId, cameraId, itemId 
                        From Temp_Bbox WHERE Temp_Bbox.itemId NOT IN ( 
                            SELECT DISTINCT tempId 
                            FROM min_join_pair)) as filtered_bbox;';
    ELSE
        EXECUTE 'INSERT INTO Materialized_Bbox (itemId, trajBbox)
                    SELECT filtered_bbox.newId, ARRAY(SELECT Temp_Bbox.trajBbox 
                                                    FROM Temp_Bbox 
                                                    WHERE Temp_Bbox.cameraId = filtered_bbox.cameraId 
                                                    AND Temp_Bbox.itemId = filtered_bbox.itemId) as new_bboxes
                    FROM (SELECT distinct CONCAT(cameraId, ' || E'\'_\'' || ', itemId) as newId, cameraId, itemId 
                        From Temp_Bbox WHERE Temp_Bbox.itemId NOT IN ( 
                            SELECT DISTINCT tempId 
                            FROM min_join_pair)) as filtered_bbox;';
    END IF;
END;
$BODY$
LANGUAGE 'plpgsql';

