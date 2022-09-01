DROP FUNCTION IF EXISTS bbox_union(stbox, stbox);
CREATE OR REPLACE FUNCTION bbox_union(stbox1 stbox, stbox2 stbox)
RETURNS stbox AS
$BODY$
DECLARE
    min_x float;
    min_y float;
    min_z float;
    min_t timestamptz;
    max_x float;
    max_y float;
    max_z float;
    max_t timestamptz;
BEGIN
    If Xmin(stbox1) < Xmin(stbox2) Then
        min_x := Xmin(stbox1);
    Else
        min_x := Xmin(stbox2);
    End If;
    If Ymin(stbox1) < Ymin(stbox2) Then
        min_y := Ymin(stbox1);
    Else
        min_y := Ymin(stbox2);
    End If;
    If Zmin(stbox1) < Zmin(stbox2) Then
        min_z := Zmin(stbox1);
    Else
        min_z := Zmin(stbox2);
    End If;
    If Tmin(stbox1) < Tmin(stbox2) Then
        min_t := Tmin(stbox1);
    Else
        min_t := Tmin(stbox2);
    End If;
    If Xmax(stbox1) > Xmax(stbox2) Then
        max_x := Xmax(stbox1);
    Else
        max_x := Xmax(stbox2);
    End If;
    If Ymax(stbox1) > Ymax(stbox2) Then
        max_y := Ymax(stbox1);
    Else
        max_y := Ymax(stbox2);
    End If;
    If Zmax(stbox1) > Zmax(stbox2) Then
        max_z := Zmax(stbox1);
    Else
        max_z := Zmax(stbox2);
    End If;
    If Tmax(stbox1) > Tmax(stbox2) Then
        max_t := Tmax(stbox1);
    Else
        max_t := Tmax(stbox2);
    End If;
    RETURN stbox(format('STBOX ZT((%s, %s, %s, %s), (%s, %s, %s, %s))', 
        min_x, min_y, min_z, min_t, max_x, max_y, max_z, max_t));
    
END;
$BODY$ 
LANGUAGE 'plpgsql';

DROP FUNCTION IF EXISTS mergeGeoArray(stbox [], stbox [], integer, integer, stbox []);
CREATE OR REPLACE FUNCTION mergeGeoArray(main_bbox stbox [], current_bbox stbox [], i integer, j integer, merged_bbox stbox []) 
RETURNS stbox [] AS 
$BODY$
DECLARE
    merged_bbox stbox [] = '{}';
    i integer = 1;
    j integer = 1;
BEGIN
    WHILE i <= array_length(main_bbox,1) and j <= array_upper(current_bbox, 1) LOOP
        IF Tmin(main_bbox[i]) <  Tmin(current_bbox[j]) THEN
            merged_bbox := array_append(merged_bbox, main_bbox[i]);
            i := i + 1;
        ELSEIF Tmin(main_bbox[i]) >  Tmin(current_bbox[j]) THEN
            merged_bbox := array_append(merged_bbox, current_bbox[j]);
            j := j + 1;
        ELSE
            merged_bbox := array_append(merged_bbox, bbox_union(main_bbox[i], current_bbox[j]));
            i := i + 1;
            j := j + 1;
        END IF;
    END LOOP;
    WHILE i <= array_length(main_bbox, 1) LOOP
        merged_bbox := array_append(merged_bbox, main_bbox[i]);
        i := i + 1;
    END LOOP;
    WHILE j <= array_length(current_bbox, 1) LOOP
        merged_bbox := array_append(merged_bbox, current_bbox[j]);
        j := j + 1;
    END LOOP;
    RETURN merged_bbox;
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
    FOR geo_camId IN EXECUTE E'SELECT DISTINCT Main_Bbox.cameraId
                                    FROM Main_Bbox
                                    WHERE Main_Bbox.itemId = \'' || $1 || E'\';'
    LOOP
        current_bbox := ARRAY(
            SELECT trajBbox 
            FROM Main_Bbox
            WHERE cameraId = geo_camId.cameraId AND Main_Bbox.itemId = $1
            );
        raise notice 'current length %', array_length(current_bbox, 1);
        IF main_bbox ISNULL THEN
            raise notice 'first bbox array';
            main_bbox := current_bbox;
        ELSE
            raise notice 'merge bbox array';
            main_bbox := mergeGeoArray(main_bbox, current_bbox, 1, 1, '{}');
        END IF;
        
    END LOOP;
    raise notice 'length %', array_length(main_bbox, 1);
  RETURN unpackBbox(main_bbox);
END;
$BODY$
LANGUAGE 'plpgsql';

