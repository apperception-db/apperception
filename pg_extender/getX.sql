DROP FUNCTION IF EXISTS getX(geometry);
CREATE OR REPLACE FUNCTION getX(p geometry) RETURNS float AS
$BODY$
BEGIN
  RETURN ST_X(p);
END
$BODY$
LANGUAGE 'plpgsql' ;

DROP FUNCTION IF EXISTS getX(geometry, timestamptz);
CREATE OR REPLACE FUNCTION getX(p geometry, t timestamptz) RETURNS float AS
$BODY$
BEGIN
  RETURN ST_X(p);
END
$BODY$
LANGUAGE 'plpgsql' ;

DROP FUNCTION IF EXISTS getX(tgeompoint, timestamptz);
CREATE OR REPLACE FUNCTION getX(tg tgeompoint, t timestamptz) RETURNS float AS
$BODY$
BEGIN
  RETURN getX(valueAtTimestamp(tg, t));
END
$BODY$
LANGUAGE 'plpgsql' ;

/* test case
 select get_X(trajcentroids, timestamptz '2021-06-08 07:10:29+00') from item_general_trajectory limit 1;
*/