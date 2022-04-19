DROP FUNCTION IF EXISTS roadDirection(geometry);
CREATE OR REPLACE FUNCTION roadDirection(point geometry) RETURNS real AS
$BODY$
BEGIN
  RETURN;
END
$BODY$
LANGUAGE 'plpgsql' ;

DROP FUNCTION IF EXISTS roadDirection(tgeompoints, timestamptz);
CREATE OR REPLACE FUNCTION roadDirection(trajectory tgeompoints, _time timestamptz) RETURNS real AS
$BODY$
BEGIN
  RETURN roadDirection(point);
END
$BODY$
LANGUAGE 'plpgsql' ;