\echo ""
\echo ""
\echo ""
\echo "facingRelative"
\echo ""

CREATE OR REPLACE FUNCTION facingRelative(object_heading real, camera_heading real) RETURNS real AS
$BODY$
BEGIN
  RETURN (((object_heading - camera_heading)::numeric % 360) + 360) % 360;
END
$BODY$
LANGUAGE 'plpgsql' ;

CREATE OR REPLACE FUNCTION facingRelative(object_heading real, camera_heading real, _time timestamptz) RETURNS real AS
$BODY$
BEGIN
  RETURN facingRelative(object_heading, camera_heading);
END
$BODY$
LANGUAGE 'plpgsql' ;

CREATE OR REPLACE FUNCTION facingRelative(object_headings tfloat, camera_heading real, _time timestamptz) RETURNS real AS
$BODY$
BEGIN
  RETURN facingRelative(CAST(valueAtTimestamp(object_headings, _time) AS real), camera_heading);
END
$BODY$
LANGUAGE 'plpgsql' ;

CREATE OR REPLACE FUNCTION facingRelative(object1_headings tfloat, object2_headings tfloat, _time timestamptz) RETURNS real AS
$BODY$
BEGIN
  RETURN facingRelative(CAST(valueAtTimestamp(object1_headings, _time) AS real), CAST(valueAtTimestamp(object2_headings, _time) AS real));
END
$BODY$
LANGUAGE 'plpgsql' ;