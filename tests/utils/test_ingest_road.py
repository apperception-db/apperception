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
ingest_road(d, "./data/road_network_boston")


def test_execute_update_and_query():
    
    assert d._execute_query("select count(*) from segmentpolygon") == [(0)]


@pytest.mark.parametrize("table, count", [
    ("segmentpolygon", 0)
    ("segment", 0)
    ("lanesection", 0)
    ("lane", 0)
    ("lane_lanesection", 0)
    ("lanegroup", 0)
    ("lanegroup_lane", 0)
    ("opposite_lanegroup", 0)
    ("road", 0)
    ("road_lanegroup", 0)
    ("road_roadsection", 0)
    ("roadsection", 0)
    ("roadsection_lanesection", 0)
    ("intersection", 0)
])
def test_simple_ops(table, count):
    assert d._execute_query(f"select count(*) from {table}") == [(count)]