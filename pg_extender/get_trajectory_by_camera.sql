DROP FUNCTION IF EXISTS getTrajectoryByCamera(text, text);
CREATE OR REPLACE FUNCTION getTrajectoryByCamera(itemId text, cameraId text) RETURNS tgeompoint AS
$BODY$
DECLARE
    interval_to_merge periodset;
    main_traj tgeompoint;
    traj_record record;
BEGIN
    raise notice 'before loop';
    FOR traj_record IN EXECUTE E'SELECT Main_Trajectory.trajCentroids
                                    FROM Main_Trajectory
                                    WHERE Main_Trajectory.itemId = \'' || $1 || E'\'' || 
                                    E'AND Main_Trajectory.cameraId = \'' || $2 || E'\';'
    LOOP
        raise notice 'inside the loop';
        IF main_traj ISNULL THEN
            main_traj := traj_record.trajCentroids;
        ELSE
            interval_to_merge := getTime(traj_record.trajCentroids) - getTime(main_traj) * getTime(traj_record.trajCentroids);
            main_traj := merge(main_traj, atPeriodSet(traj_record.trajCentroids, interval_to_merge));
        END IF;
        
    END LOOP;
  RETURN main_traj;
END;
$BODY$
LANGUAGE 'plpgsql';

