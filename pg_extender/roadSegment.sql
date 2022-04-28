CREATE OR REPLACE FUNCTION roadSegment(segment_type text) RETURNS geometry[] AS
$BODY$
DECLARE
 segment_polygons geometry[];
BEGIN
    EXECUTE format('SELECT ARRAY(SELECT SegmentPolygon.elementPolygon FROM SegmentPolygon, %I 
    WHERE SegmentPolygon.elementId = %I.id)', segment_type, segment_type) into segment_polygons;
    RETURN segment_polygons;
END
$BODY$
LANGUAGE 'plpgsql' ;