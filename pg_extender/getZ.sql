\echo ""
\echo ""
\echo ""
\echo "getZ"
\echo ""

DROP FUNCTION IF EXISTS getZ(geometry);
CREATE OR REPLACE FUNCTION getZ(p geometry) RETURNS float AS
$BODY$
BEGIN
  RETURN ST_Z(p);
END
$BODY$
LANGUAGE 'plpgsql' ;

DROP FUNCTION IF EXISTS getY(geometry, timestamptz);
CREATE OR REPLACE FUNCTION getZ(p geometry, t timestamptz) RETURNS float AS
$BODY$
BEGIN
  RETURN ST_Z(p);
END
$BODY$
LANGUAGE 'plpgsql' ;

DROP FUNCTION IF EXISTS getZ(tgeompoint, timestamptz);
CREATE OR REPLACE FUNCTION getZ(tg tgeompoint, t timestamptz) RETURNS float AS
$BODY$
BEGIN
  RETURN getZ(valueAtTimestamp(tg, t));
END
$BODY$
LANGUAGE 'plpgsql' ;

/* test case
 select get_Z(trajcentroids, timestamptz '2021-06-08 07:10:29+00') from item_general_trajectory limit 1;
*/