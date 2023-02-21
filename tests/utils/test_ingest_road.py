from apperception.database import Database
from apperception.utils import ingest_road
import psycopg2
import os
import pytest


d = Database(psycopg2.connect(
    dbname="mobilitydb",
    user="docker",
    host="localhost",
    port=os.environ["AP_PORT_ROAD"],
    password="docker",
))
ingest_road(d, "./data/scenic/road_network_boston")


@pytest.mark.parametrize("table, count", [
    ("segmentpolygon", 3067),
    ("segment", 11379),
    ("lanesection", 1178),
    ("lane", 1178),
    ("lane_lanesection", 1178),
    ("lanegroup", 964),
    ("lanegroup_lane", 1178),
    ("opposite_lanegroup", 742),
    ("road", 925),
    ("road_lanegroup", 964),
    ("road_roadsection", 593),
    ("roadsection", 593),
    ("roadsection_lanesection", 1178),
    ("intersection", 332),
])
def test_simple_ops(table, count):
    assert d.execute(f"select count(*) from {table}") == [(count,)]