\echo ""
\echo ""
\echo ""
\echo "geoContained"
\echo ""

/* return true if a point is contained in any of the geometry in the array*/

/* this first function should already exist in mobilitydb, plz double check**/
DROP FUNCTION IF EXISTS contained(geometry, geometry); 
CREATE OR REPLACE FUNCTION contained(contPoint geometry, geom geometry) RETURNS boolean AS
$BODY$
BEGIN
  RETURN ST_Covers(geom, contPoint);
END
$BODY$
LANGUAGE 'plpgsql' ;

DROP FUNCTION IF EXISTS contained(geometry, geometry[]);
CREATE OR REPLACE FUNCTION contained(contPoint geometry, geoms geometry[]) RETURNS boolean AS
$BODY$
declare geom geometry;
BEGIN
  FOREACH geom IN ARRAY geoms
  LOOP
    IF contained(contPoint, geom) THEN
      RETURN true;
    END IF;  
  END LOOP;
  RETURN false; 
END
$BODY$
LANGUAGE 'plpgsql' ;

DROP FUNCTION IF EXISTS contained(tgeompoint, geometry[], timestamptz);
CREATE OR REPLACE FUNCTION contained(contPoint tgeompoint, geoms geometry[], t timestamptz) RETURNS boolean AS
$BODY$
BEGIN
  RETURN contained(valueAtTimestamp(contPoint, t), geoms);
END
$BODY$
LANGUAGE 'plpgsql' ;