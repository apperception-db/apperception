DROP FUNCTION IF EXISTS facingRelative(real, real);
CREATE OR REPLACE FUNCTION facingRelative(object_heading real, camera_heading real) RETURNS real AS
$BODY$
BEGIN
  RETURN object_heading - camera_heading;
END
$BODY$
LANGUAGE 'plpgsql' ;


DROP FUNCTION IF EXISTS facingRelative(tfloat, real, timestamptz);
CREATE OR REPLACE FUNCTION facingRelative(object_headings tfloat, camera_heading real, _time timestamptz) RETURNS real AS
$BODY$
BEGIN
  RETURN facingRelative(CAST(valueAtTimestamp(object_headings, _time) AS real), camera_heading);
END
$BODY$
LANGUAGE 'plpgsql' ;