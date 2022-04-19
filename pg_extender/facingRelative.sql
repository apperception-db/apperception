DROP FUNCTION IF EXISTS facingRelative(real, real, timestamptz);
CREATE OR REPLACE FUNCTION facingRelative(object_heading real, camera_heading real, _time timestamptz) RETURNS real AS
$BODY$
BEGIN
  #TODO: RETURN ;
END
$BODY$
LANGUAGE 'plpgsql' ;


DROP FUNCTION IF EXISTS facingRelative(tfloat, real, timestamptz);
CREATE OR REPLACE FUNCTION facingRelative(object_headings tfloat, camera_heading real, _time timestamptz) RETURNS real AS
$BODY$
BEGIN
  # TODO: get object heading at timestampz
  RETURN facingRelative(object_heading, camera_heading, _time);
END
$BODY$
LANGUAGE 'plpgsql' ;