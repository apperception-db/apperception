DROP FUNCTION IF EXISTS roadDirection(real, real);
CREATE OR REPLACE FUNCTION roadDirection(x real, y real) RETURNS real AS
$BODY$
BEGIN
  RETURN (SELECT heading FROM segment, st_point(x, y) AS POINT, 
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
  RETURN roadDirection(ST_X(ST_Centroid(cordPoint)), ST_Y(ST_Centroid(cordPoint)));
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