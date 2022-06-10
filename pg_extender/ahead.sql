\echo ""
\echo ""
\echo ""
\echo "ahead"
\echo ""

DROP FUNCTION IF EXISTS ahead(geometry, geometry, real);
CREATE OR REPLACE FUNCTION ahead(obj1_loc geometry, obj2_loc geometry, obj2_heading real) RETURNS boolean AS
$BODY$
BEGIN
  RETURN (ST_X(obj1_loc) - ST_X(obj2_loc)) * COS(PI() * (obj2_heading) / 180) + (ST_Y(obj1_loc) - ST_Y(obj2_loc)) * SIN(PI() * (obj2_heading) / 180) > 0 AND
         ABS(ST_X(obj1_loc) - ST_X(obj2_loc)) < 1;
END
$BODY$
LANGUAGE 'plpgsql' ;

DROP FUNCTION IF EXISTS ahead(geometry, geometry, real, timestamptz);
CREATE OR REPLACE FUNCTION ahead(obj1_loc geometry, obj2_loc geometry, obj2_heading real, t timestamptz) RETURNS boolean AS
$BODY$
BEGIN
  RETURN ahead(obj1_loc, obj2_loc, obj2_heading);
END
$BODY$
LANGUAGE 'plpgsql' ;

DROP FUNCTION IF EXISTS ahead(geometry, tgeompoint, tfloat, timestamptz);
CREATE OR REPLACE FUNCTION ahead(obj1_loc geometry, obj2_loc tgeompoint, obj2_heading tfloat, t timestamptz) RETURNS boolean AS
$BODY$
BEGIN
  RETURN ahead(obj1_loc, valueAtTimestamp(obj2_loc, t), CAST(valueAtTimestamp(object2_headin, t) AS real));
END
$BODY$
LANGUAGE 'plpgsql' ;

DROP FUNCTION IF EXISTS ahead(tgeompoint, geometry, real, timestamptz);
CREATE OR REPLACE FUNCTION ahead(obj1_loc tgeompoint, obj2_loc geometry, obj2_heading real, t timestamptz) RETURNS boolean AS
$BODY$
BEGIN
  RETURN ahead(valueAtTimestamp(obj1_loc, t), obj2_loc, obj2_heading);
END
$BODY$
LANGUAGE 'plpgsql' ;

DROP FUNCTION IF EXISTS ahead(tgeompoint, tgeompoint, tfloat, timestamptz);
CREATE OR REPLACE FUNCTION ahead(obj1_loc tgeompoint, obj2_loc tgeompoint, obj2_heading tfloat, t timestamptz) RETURNS boolean AS
$BODY$
BEGIN
  RETURN ahead(valueAtTimestamp(obj1_loc, t), valueAtTimestamp(obj2_loc, t), CAST(valueAtTimestamp(object2_headin, t) AS real));
END
$BODY$
LANGUAGE 'plpgsql' ;