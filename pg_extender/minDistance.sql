\echo ""
\echo ""
\echo ""
\echo "minDistance"
\echo ""

DROP FUNCTION IF EXISTS minDistance(geometry, geometry[]);
CREATE OR REPLACE FUNCTION minDistance(p geometry, geoms geometry[]) RETURNS real AS
$BODY$
declare geom geometry;
declare min_dis real;
BEGIN
    min_dis := '+infinity'::real;
    FOREACH geom IN ARRAY geoms
    LOOP
        min_dis := LEAST(min_dis, ST_Distance(p, geom));
    END LOOP;
    RETURN min_dis;
END
$BODY$
LANGUAGE 'plpgsql' ;