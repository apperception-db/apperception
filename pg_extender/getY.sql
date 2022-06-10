\echo ""
\echo ""
\echo ""
\echo "getY"
\echo ""

DROP FUNCTION IF EXISTS getY(geometry);
CREATE OR REPLACE FUNCTION getY(p geometry) RETURNS float AS
$BODY$
BEGIN
  RETURN ST_Y(p);
END
$BODY$
LANGUAGE 'plpgsql' ;

DROP FUNCTION IF EXISTS getY(geometry, timestamptz);
CREATE OR REPLACE FUNCTION getY(p geometry, t timestamptz) RETURNS float AS
$BODY$
BEGIN
  RETURN ST_Y(p);
END
$BODY$
LANGUAGE 'plpgsql' ;

DROP FUNCTION IF EXISTS getY(tgeompoint, timestamptz);
CREATE OR REPLACE FUNCTION getY(tg tgeompoint, t timestamptz) RETURNS float AS
$BODY$
BEGIN
  RETURN getY(valueAtTimestamp(tg, t));
END
$BODY$
LANGUAGE 'plpgsql' ;

/* test case
 select get_Y(trajcentroids, timestamptz '2021-06-08 07:10:29+00') from item_general_trajectory limit 1;
*/