from apperception.database import database

res = database.execute('select elementid, elementpolygon, segmenttypes from segmentpolygon limit 1')
print(res)

res = database.execute('select segmentline, heading from segment limit 1')
print(res)