DROP FUNCTION IF EXISTS createAllTimeIntersectView();
CREATE OR REPLACE FUNCTION createAllTimeIntersectView() RETURNS void AS
$BODY$
BEGIN
  EXECUTE 'CREATE OR REPLACE VIEW all_time_intersect AS  
            SELECT Main_Trajectory.itemId AS mainId,  
                  Temp_Trajectory.itemId AS tempId,  
                  getTime(Main_Trajectory.trajCentroids) * getTime(Temp_Trajectory.trajCentroids) AS intersection 
              FROM Main_Trajectory, Temp_Trajectory, Item_Meta  
              WHERE Main_Trajectory.itemId = Item_Meta.itemId  
                AND Item_Meta.objectType = Temp_Trajectory.objectType;';
  RETURN;
END;
$BODY$
LANGUAGE 'plpgsql';

DROP FUNCTION IF EXISTS createAllPairOfDistance();
CREATE OR REPLACE FUNCTION createAllPairOfDistance() RETURNS void AS
$BODY$
BEGIN
  EXECUTE 'CREATE OR REPLACE VIEW all_pair_of_distance AS  
              SELECT all_time_intersect.mainId, 
                     all_time_intersect.tempId,  
                     nearestApproachDistance(atPeriodSet(Main_Trajectory.trajCentroids, intersection),  
                                             atPeriodSet(Temp_Trajectory.trajCentroids, intersection)) 
                FROM Main_Trajectory, Temp_Trajectory, all_time_intersect  
                WHERE Main_Trajectory.itemId = all_time_intersect.mainId  
                  AND Temp_Trajectory.itemId = all_time_intersect.tempId 
                  AND numTimestamps(all_time_intersect.intersection) > 0;';
  RETURN;
END;
$BODY$
LANGUAGE 'plpgsql';

DROP FUNCTION IF EXISTS reconcileTrajectory(boolean, int);
CREATE OR REPLACE FUNCTION reconcileTrajectory(reconcile_on_id boolean, threshold int) RETURNS void AS
$BODY$
BEGIN
  IF NOT reconcile_on_id then
    EXECUTE 'SELECT createAllTimeIntersectView();';

    EXECUTE 'SELECT createAllPairOfDistance();';

  -- Create the pairs of trajectory to join
    EXECUTE 'CREATE OR REPLACE VIEW join_pair AS  
              SELECT all_pair_of_distance.mainId, tempId, min_distance.min  
                FROM all_pair_of_distance,  
                    (SELECT mainid, Min(nearestapproachdistance) FROM all_pair_of_distance GROUP BY mainId) AS min_distance  
                WHERE all_pair_of_distance.nearestapproachdistance = min_distance.min  
                  AND min_distance.min < ' || $2 || ';'; 

  -- Filter out the best pairs
    EXECUTE 'CREATE TABLE min_join_pair AS
              SELECT join_pair.mainId, join_pair.tempId 
                FROM join_pair, 
                    (SELECT tempId, Min(join_pair.min) FROM join_pair GROUP BY tempid) as min_pair 
                WHERE join_pair.tempId = min_pair.tempId AND join_pair.min = min_pair.min;';
  ELSE
    EXECUTE 'CREATE TABLE min_join_pair AS
              SELECT Distinct Main_Trajectory.itemId AS mainId, Temp_Trajectory.itemId AS tempId 
                FROM Main_Trajectory, Temp_Trajectory
                WHERE Main_Trajectory.itemId = Temp_Trajectory.itemId;';
  END IF;

-- UPDATE Trajectory Tables
  EXECUTE 'INSERT INTO Main_Trajectory  
            SELECT min_join_pair.mainId, Temp_Trajectory.cameraId, Temp_Trajectory.trajCentroids  
              FROM min_join_pair, Temp_Trajectory  
              WHERE min_join_pair.tempId = Temp_Trajectory.itemId;';
  
  EXECUTE 'SELECT materializeExistingTrajectory(min_join_pair.mainId, Temp_Trajectory.trajCentroids)
            FROM min_join_pair, Temp_Trajectory  
              WHERE min_join_pair.tempId = Temp_Trajectory.itemId;';

  EXECUTE 'INSERT INTO Main_Bbox  
            SELECT min_join_pair.mainId, Temp_Bbox.cameraId, Temp_Bbox.trajBbox  
              FROM min_join_pair, Temp_Bbox  
              WHERE min_join_pair.tempId = Temp_Bbox.itemId;';

  EXECUTE 'SELECT  materializeExistingBbox(filtered_bbox.mainId, 
                                          ARRAY(
                                            SELECT Temp_Bbox.trajBbox 
                                            FROM Temp_Bbox, min_join_pair
                                            WHERE Temp_Bbox.itemId = min_join_pair.tempId 
                                            and min_join_pair.mainId = filtered_bbox.mainId))
            FROM (SELECT distinct min_join_pair.mainId FROM min_join_pair, Temp_Bbox WHERE min_join_pair.tempId = Temp_Bbox.itemId) as filtered_bbox;';

  IF reconcile_on_id THEN
    EXECUTE 'INSERT INTO MAIN_Trajectory 
              SELECT Temp_Trajectory.itemId, 
                    Temp_Trajectory.cameraId, Temp_Trajectory.trajCentroids  
                FROM Temp_Trajectory  
                WHERE Temp_Trajectory.itemId NOT IN ( 
                  SELECT DISTINCT tempId 
                  FROM min_join_pair);';
  ELSE
    EXECUTE 'INSERT INTO MAIN_Trajectory 
              SELECT CONCAT(Temp_Trajectory.cameraId, ' || E'\'_\'' || ', 
                            Temp_Trajectory.itemId), 
                    Temp_Trajectory.cameraId, Temp_Trajectory.trajCentroids  
                FROM Temp_Trajectory  
                WHERE Temp_Trajectory.itemId NOT IN ( 
                  SELECT DISTINCT tempId 
                  FROM min_join_pair);';
  END IF;
  
  IF reconcile_on_id THEN
    EXECUTE 'SELECT materializeNewTrajectory(Temp_Trajectory.itemId,  Temp_Trajectory.trajCentroids)
              FROM Temp_Trajectory  
                WHERE Temp_Trajectory.itemId NOT IN ( 
                  SELECT DISTINCT tempId 
                  FROM min_join_pair);';
  ELSE
    EXECUTE 'SELECT materializeNewTrajectory(CONCAT(Temp_Trajectory.cameraId, ' || E'\'_\'' || ', Temp_Trajectory.itemId),  Temp_Trajectory.trajCentroids)
              FROM Temp_Trajectory  
                WHERE Temp_Trajectory.itemId NOT IN ( 
                  SELECT DISTINCT tempId 
                  FROM min_join_pair);';
  END IF;

  IF reconcile_on_id THEN
    EXECUTE 'INSERT INTO MAIN_Bbox 
            SELECT Temp_Bbox.itemId, Temp_Bbox.cameraId, Temp_Bbox.trajBbox  
              FROM Temp_Bbox  
              WHERE Temp_Bbox.itemId NOT IN ( 
                SELECT DISTINCT tempId 
                FROM min_join_pair);';
  ELSE
    EXECUTE 'INSERT INTO MAIN_Bbox 
              SELECT CONCAT(Temp_Bbox.cameraId, ' || E'\'_\'' || ', Temp_Bbox.itemId), Temp_Bbox.cameraId, Temp_Bbox.trajBbox  
                FROM Temp_Bbox  
                WHERE Temp_Bbox.itemId NOT IN ( 
                  SELECT DISTINCT tempId 
                  FROM min_join_pair);';
  END IF;

  IF reconcile_on_id THEN
    EXECUTE 'SELECT materializeNewBbox(TRUE);';
  ELSE
    EXECUTE 'SELECT materializeNewBbox(FALSE);';
  END IF;

  IF reconcile_on_id THEN
    EXECUTE 'INSERT INTO Item_Meta  
                SELECT Temp_Trajectory.itemId, Temp_Trajectory.objectType, 
                       Temp_Trajectory.color, Temp_Trajectory.largestBbox 
                  FROM Temp_Trajectory  
                  WHERE Temp_Trajectory.itemId NOT IN ( 
                    SELECT DISTINCT tempId 
                    FROM min_join_pair);';
  ELSE
    EXECUTE 'INSERT INTO Item_Meta  
              SELECT CONCAT(Temp_Trajectory.cameraId, ' || E'\'_\'' || ', Temp_Trajectory.itemId), 
                    Temp_Trajectory.objectType, Temp_Trajectory.color, Temp_Trajectory.largestBbox 
                FROM Temp_Trajectory  
                WHERE Temp_Trajectory.itemId NOT IN ( 
                  SELECT DISTINCT tempId 
                  FROM min_join_pair);';
  END IF;

  EXECUTE 'TRUNCATE TABLE Temp_Trajectory;';
  EXECUTE 'TRUNCATE TABLE Temp_Bbox;';
  EXECUTE 'DROP TABLE min_join_pair;';

  RETURN;
END;
$BODY$
LANGUAGE 'plpgsql';