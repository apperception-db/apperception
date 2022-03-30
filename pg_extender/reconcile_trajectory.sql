DROP FUNCTION IF EXISTS create_all_time_intersect_view();
CREATE OR REPLACE FUNCTION create_all_time_intersect_view() RETURNS void AS
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

DROP FUNCTION IF EXISTS create_all_pair_of_distance();
CREATE OR REPLACE FUNCTION create_all_pair_of_distance() RETURNS void AS
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
END
$BODY$
LANGUAGE 'plpgsql';

DROP FUNCTION IF EXISTS reconcile_trajectory(int);
CREATE OR REPLACE FUNCTION reconcile_trajectory(threshold int) RETURNS void AS
$BODY$
BEGIN
  EXECUTE 'SELECT create_all_time_intersect_view();';

  EXECUTE 'SELECT create_all_pair_of_distance();';
-- Create the pairs of trajectory to join
  EXECUTE 'CREATE OR REPLACE VIEW join_pair AS  
            SELECT all_pair_of_distance.mainId, tempId, min_distance.min  
              FROM all_pair_of_distance,  
                   (SELECT mainid, Min(nearestapproachdistance) FROM all_pair_of_distance GROUP BY mainId) AS min_distance  
              WHERE all_pair_of_distance.nearestapproachdistance = min_distance.min  
                AND min_distance.min < ' || $1 || ';'; 
-- Filter out the best pairs
  EXECUTE 'CREATE OR REPLACE VIEW min_join_pair AS
            SELECT join_pair.mainId, join_pair.tempId 
              FROM join_pair, 
                  (SELECT tempId, Min(join_pair.min) FROM join_pair GROUP BY tempid) as min_pair 
              WHERE join_pair.tempId = min_pair.tempId AND join_pair.min = min_pair.min;';
-- UPDATE Trajectory Tables
  EXECUTE 'INSERT INTO Main_Trajectory  
            SELECT min_join_pair.mainId, Temp_Trajectory.cameraId, Temp_Trajectory.trajCentroids  
              FROM min_join_pair, Temp_Trajectory  
              WHERE min_join_pair.tempId = Temp_Trajectory.itemId;';

  EXECUTE 'INSERT INTO Main_Bbox  
            SELECT min_join_pair.mainId, Temp_Bbox.cameraId, Temp_Bbox.trajBbox  
              FROM min_join_pair, Temp_Bbox  
              WHERE min_join_pair.tempId = Temp_Bbox.itemId;';

  EXECUTE 'INSERT INTO MAIN_Trajectory 
            SELECT CONCAT(Temp_Trajectory.cameraId, ' || E'\'_\'' || ', Temp_Trajectory.itemId), Temp_Trajectory.cameraId, Temp_Trajectory.trajCentroids  
              FROM Temp_Trajectory  
              WHERE Temp_Trajectory.itemId NOT IN ( 
                SELECT DISTINCT tempId 
                FROM min_join_pair);';
  
  EXECUTE 'INSERT INTO MAIN_Bbox 
            SELECT CONCAT(Temp_Bbox.cameraId, ' || E'\'_\'' || ', Temp_Bbox.itemId), Temp_Bbox.cameraId, Temp_Bbox.trajBbox  
              FROM Temp_Bbox  
              WHERE Temp_Bbox.itemId NOT IN ( 
                SELECT DISTINCT tempId 
                FROM min_join_pair);';

  EXECUTE 'INSERT INTO Item_Meta  
            SELECT CONCAT(Temp_Trajectory.cameraId, ' || E'\'_\'' || ', Temp_Trajectory.itemId), 
                   Temp_Trajectory.objectType, Temp_Trajectory.color, Temp_Trajectory.largestBbox 
              FROM Temp_Trajectory  
              WHERE Temp_Trajectory.itemId NOT IN ( 
                SELECT DISTINCT tempId 
                FROM min_join_pair);';
  EXECUTE 'TRUNCATE TABLE Temp_Trajectory;';
  EXECUTE 'TRUNCATE TABLE Temp_Bbox;'

  RETURN;
END
$BODY$
LANGUAGE 'plpgsql' ;