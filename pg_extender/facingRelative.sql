DROP FUNCTION IF EXISTS facingRelative(real, real, timestamptz);
CREATE OR REPLACE FUNCTION facingRelative(object_heading real, camera_heading real, _time timestamptz) RETURNS real AS
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
  -- TODO: get object heading at timestampz
  RETURN facingRelative(valueAtTimestamp(object_headings, _time), camera_heading, _time);
END
$BODY$
LANGUAGE 'plpgsql' ;