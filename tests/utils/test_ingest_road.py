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
ingest_road(d, "./data/scenic/road-network")


@pytest.mark.parametrize("table, count", [
    ("segmentpolygon", 3072),
    ("segment", 11410),
    ("lanesection", 1180),
    ("lane", 1180),
    ("lane_lanesection", 1180),
    ("lanegroup", 966),
    ("lanegroup_lane", 1180),
    ("opposite_lanegroup", 744),
    ("road", 926),
    ("road_lanegroup", 966),
    ("road_roadsection", 594),
    ("roadsection", 594),
    ("roadsection_lanesection", 1180),
    ("intersection", 332),
])
def test_simple_ops(table, count):
    assert d.execute(f"select count(*) from {table}") == [(count,)]