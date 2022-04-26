DROP FUNCTION IF EXISTS ConvertCamera(geometry, geometry, real);
CREATE OR REPLACE FUNCTION ConvertCamera(objPoint geometry, camPoint geometry, camHeading real) RETURNS geometry AS
$BODY$
DECLARE subtract_x real;
DECLARE subtract_y real;
DECLARE subtract_mag real;

BEGIN
  subtract_x := ST_X(ST_Centroid(objPoint)) - ST_X(ST_Centroid(camPoint));
  subtract_y := ST_Y(ST_Centroid(objPoint)) - ST_Y(ST_Centroid(camPoint));
  subtract_mag := SQRT(POWER(subtract_x, 2) + POWER(subtract_y, 2));
  RETURN ST_MakePoint(subtract_mag * COS(PI() * camHeading / 180 + ATAN2(subtract_y, subtract_x)),
                      subtract_mag * SIN(PI() * camHeading / 180 + ATAN2(subtract_y, subtract_x)));
END
$BODY$
LANGUAGE 'plpgsql';

DROP FUNCTION IF EXISTS ConvertCamera(tgeompoint, geometry, real, timestamptz);
CREATE OR REPLACE FUNCTION ConvertCamera(traj tgeompoint, camPoint geometry, camHeading real, t timestamptz) RETURNS geometry AS
$BODY$
BEGIN
  RETURN ConvertCamera(valueAtTimestamp(traj, t), camPoint, camHeading);
END
$BODY$
LANGUAGE 'plpgsql';