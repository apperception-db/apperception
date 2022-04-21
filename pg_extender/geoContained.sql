/* return true if a point is contained in any of the geometry in the array*/

/* this first function should already exist in mobilitydb, plz double check**/
DROP FUNCTION IF EXISTS contained(geometry, geometry); 

DROP FUNCTION IF EXISTS contained(geometry, geometry[]);

DROP FUNCTION IF EXISTS contained(tgeompoint, geometry[],timestamptz)