-- Checks whether a segment of the specified type is in view within some distance and some view angle
DROP FUNCTION IF EXISTS inView(text, real, geometry, real, real);
CREATE OR REPLACE FUNCTION inView(segment_type text, view_point_heading real, view_point geometry, max_distance real, view_angle real) RETURNS boolean AS 
$BODY$
declare clockwise numeric;
declare counterClockwise numeric;
declare azimuth numeric;
-- Note: view_point_heading is counter-clockwise with North being 0, while the result of ST_Azimuth is clockwise with North being 0
BEGIN

    RETURN EXISTS(SELECT * FROM SegmentPolygon 
                        WHERE segment_type = Any(segmentTypes) 
                          AND ST_Distance(view_point, elementPolygon) < max_distance 
                          AND viewAngle(ST_Centroid(elementPolygon) , view_point_heading, view_point) < view_angle);
END
$BODY$
LANGUAGE 'plpgsql' ;