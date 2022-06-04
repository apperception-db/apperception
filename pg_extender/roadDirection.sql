\echo ""
\echo ""
\echo ""
\echo "roadDirection"
\echo ""

DROP FUNCTION IF EXISTS roadDirection(x real, y real);
CREATE OR REPLACE FUNCTION roadDirection(x real, y real) RETURNS real AS
$BODY$
declare contPolygonElementId text;
declare result real;
BEGIN
     SELECT elementId INTO contPolygonElementId FROM SegmentPolygon WHERE contained(st_point(x, y), elementPolygon);
     IF contPolygonElementId IS NOT NULL THEN
        result := (SELECT heading * 180 / PI() FROM segment, st_point(x, y) AS point 
                          WHERE elementId IN (SELECT s.elementId FROM SegmentPolygon AS s WHERE contained(st_point(x, y), s.elementPolygon)) 
                                AND ROUND(CAST(heading * 180 / PI() AS numeric), 3) != -45
                          ORDER BY segmentLine <-> point ASC LIMIT 1);
        IF result IS NULL THEN
          result := (SELECT heading * 180 / PI() FROM segment, st_point(x, y) AS point 
                            WHERE ROUND(CAST(heading * 180 / PI() AS numeric), 3) != -45 
                            ORDER BY segmentLine <-> point ASC LIMIT 1 );
          -- result := (SELECT heading * 180 / PI() AS degHeading FROM segment, st_point(x, y) AS point 
          --                   WHERE elementId = contPolygonElementId
          --                   ORDER BY segmentLine <-> point ASC LIMIT 1);
        END IF;
     ELSE
        result := (SELECT heading * 180 / PI() FROM segment, st_point(x, y) AS point 
                          WHERE ROUND(CAST(heading * 180 / PI() AS numeric), 3) != -45 
                          ORDER BY segmentLine <-> point ASC LIMIT 1 );
     END IF;
     RETURN CAST(result AS numeric) % 360;
END
$BODY$
LANGUAGE 'plpgsql' ;

DROP FUNCTION IF EXISTS roadDirection(cordPoint geometry);
CREATE OR REPLACE FUNCTION roadDirection(cordPoint geometry) RETURNS real AS
$BODY$
BEGIN
  RETURN roadDirection(Cast(ST_X(ST_Centroid(cordPoint)) AS real), Cast(ST_Y(ST_Centroid(cordPoint)) AS real));
END
$BODY$
LANGUAGE 'plpgsql' ;

DROP FUNCTION IF EXISTS roadDirection(cordPoint geometry, _time timestamptz);
CREATE OR REPLACE FUNCTION roadDirection(cordPoint geometry, _time timestamptz) RETURNS real AS
$BODY$
BEGIN
  RETURN roadDirection(cordPoint);
END
$BODY$
LANGUAGE 'plpgsql' ;

DROP FUNCTION IF EXISTS roadDirection(trajectory tgeompoint, _time timestamptz);
CREATE OR REPLACE FUNCTION roadDirection(trajectory tgeompoint, _time timestamptz) RETURNS real AS
$BODY$
BEGIN
  RETURN roadDirection(valueAtTimestamp(trajectory, _time));
END
$BODY$
LANGUAGE 'plpgsql' ;