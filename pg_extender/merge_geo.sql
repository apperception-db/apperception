DROP FUNCTION IF EXISTS mergeGeoArray(stbox [], stbox [], integer, integer, stbox []);
CREATE OR REPLACE FUNCTION mergeGeoArray(main_bbox stbox [], current_bbox stbox [], i integer, j integer, merged_bbox stbox []) 
RETURNS stbox [] AS 
$BODY$
BEGIN
    SELECT 
        CASE WHEN (i > array_length(main_bbox,1) and j > array_length(current_bbox,1)) THEN merged_bbox
             WHEN i > array_length(main_bbox,1) THEN mergeGeoArray(main_bbox,current_bbox,i,j+1,array_append(merged_bbox,current_bbox[j]))
             WHEN j > array_length(current_bbox,1) THEN mergeGeoArray(main_bbox,current_bbox,i+1,j,array_append(merged_bbox,main_bbox[i]))
             WHEN Tmin(main_bbox[i]) <  Tmin(current_bbox[j]) THEN mergeGeoArray(main_bbox,current_bbox,i+1,j,array_append(merged_bbox, main_bbox[i]))
             WHEN Tmin(main_bbox[i]) > Tmin(current_bbox[j]) THEN mergeGeoArray(main_bbox,current_bbox,i,j+1,array_append(merged_bbox, current_bbox[j]))
             WHEN Tmin(main_bbox[i]) = Tmin(current_bbox[j]) THEN mergeGeoArray(main_bbox, current_bbox,i+1,j+1,array_append(merged_bbox, main_bbox[i]+main_bbox[j]))
        END;
END;
$BODY$ 
LANGUAGE 'plpgsql';

DROP FUNCTION IF EXISTS unpackBbox(stbox[]);
CREATE OR REPLACE FUNCTION unpackBbox(merged_bbox stbox[])
RETURNS float[][]
AS
$BODY$
DECLARE
   min_x float [] = '{}';
   min_y float [] = '{}';
   min_z float [] = '{}';
   max_x float [] = '{}';
   max_y float [] = '{}';
   max_z float [] = '{}';
   bbox stbox;
BEGIN
   FOREACH bbox IN ARRAY merged_bbox LOOP
    min_x := array_append(min_x, Xmin(bbox));
    min_y := array_append(min_y, Ymin(bbox));
    min_z := array_append(min_z, Zmin(bbox));
    max_x := array_append(max_x, Xmax(bbox));
    max_y := array_append(max_y, Ymax(bbox));
    max_z := array_append(max_z, Zmax(bbox));
   END LOOP;
   
RETURN ARRAY[min_x, min_y, min_z, max_x, max_y, max_z];
END;
$BODY$
LANGUAGE plpgsql;

DROP FUNCTION IF EXISTS mergeGeo(text);
CREATE OR REPLACE FUNCTION mergeGeo(itemId text) RETURNS float[][] AS
$BODY$
DECLARE
    main_bbox stbox [];
    current_bbox stbox [];
    geo_camId record;
BEGIN
    raise notice 'before loop';
    FOR geo_camId IN EXECUTE E'SELECT DISTINCT Main_Bbox.cameraId
                                    FROM Main_Bbox
                                    WHERE Main_Bbox.itemId = \'' || $1 || E'\';'
    LOOP
        raise notice 'inside the loop';
        current_bbox := ARRAY(
            SELECT trajBbox 
            FROM Main_Bbox
            WHERE cameraId = geo_camId.cameraId
            );
        IF main_bbox ISNULL THEN
            main_bbox := current_bbox;
        ELSE
            main_bbox := mergeGeoArray(main_bbox, current_bbox, 0, 0, '{}');
        END IF;
        
    END LOOP;
  RETURN unpackBbox(main_bbox);
END;
$BODY$
LANGUAGE 'plpgsql';

