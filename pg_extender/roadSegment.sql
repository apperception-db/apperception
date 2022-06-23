\echo ""
\echo ""
\echo ""
\echo "roadSegment"
\echo ""

CREATE OR REPLACE FUNCTION roadSegment(segment_type text) RETURNS geometry[] AS
$BODY$
DECLARE
 segment_polygons geometry[];
BEGIN
    IF segment_type = 'laneWithRightLane' THEN
        EXECUTE format('SELECT ARRAY(SELECT polygon1.elementPolygon FROM segmentPolygon as polygon1, 
        lane as l1, lane as l2, segmentPolygon as polygon2 
        WHERE polygon1.elementId = l1.id AND l1.id != l2.id AND polygon2.elementId = l2.id 
        AND laneToRight(polygon2.elementPolygon, polygon1.elementPolygon))') into segment_polygons;
    ELSE
        EXECUTE format('SELECT ARRAY(SELECT SegmentPolygon.elementPolygon FROM SegmentPolygon, %I 
        WHERE SegmentPolygon.elementId = %I.id)', segment_type, segment_type) into segment_polygons;
    END IF;
    RETURN segment_polygons;
END
$BODY$
LANGUAGE 'plpgsql';

-- return true, if lane_polygon1 is to the right of lane_polygon2
CREATE OR REPLACE FUNCTION laneToRight(lane_polygon1 geometry, lane_polygon2 geometry) RETURNS boolean AS
$BODY$
BEGIN
    RETURN TRUE;
END
$BODY$
LANGUAGE 'plpgsql';