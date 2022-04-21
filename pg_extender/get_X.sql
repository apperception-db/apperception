CREATE OR REPLACE FUNCTION get_X(tg tgeompoint, t timestamptz) RETURNS float AS
$BODY$
BEGIN
  RETURN ST_X(valueAtTimestamp(tg, t));
END
$BODY$
LANGUAGE 'plpgsql' ;

/* test case
/* select get_X(trajcentroids, timestamptz '2021-06-08 07:10:29+00') from item_general_trajectory limit 1;
