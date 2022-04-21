DROP FUNCTION IF EXISTS roadDirection(real, real);
CREATE OR REPLACE FUNCTION roadDirection(x real, y real) RETURNS real AS
$BODY$
BEGIN
  RETURN (SELECT heading * 180 / PI() FROM segment, st_point(x, y) AS POINT, 
                                       st_distance(st_makeline(startPoint, endPoint), point) as dis
            ORDER BY dis ASC
            LIMIT 1);
END
$BODY$
LANGUAGE 'plpgsql' ;

DROP FUNCTION IF EXISTS roadDirection(geometry);
CREATE OR REPLACE FUNCTION roadDirection(cordPoint geometry) RETURNS real AS
$BODY$
BEGIN
  RETURN roadDirection(Cast(ST_X(ST_Centroid(cordPoint)) AS real), Cast(ST_Y(ST_Centroid(cordPoint)) AS real));
END
$BODY$
LANGUAGE 'plpgsql' ;

DROP FUNCTION IF EXISTS roadDirection(tgeompoint, timestamptz);
CREATE OR REPLACE FUNCTION roadDirection(trajectory tgeompoint, _time timestamptz) RETURNS real AS
$BODY$
BEGIN
  RETURN roadDirection(valueAtTimestamp(trajectory, _time));
END
$BODY$
LANGUAGE 'plpgsql' ;