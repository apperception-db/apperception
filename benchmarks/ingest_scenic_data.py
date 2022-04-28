import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "apperception"))
import psycopg2
import argparse
from apperception.utils.import_or_export_to_db import *
from apperception.utils.road_ingest import *
import json

def ingest_data(remote: bool = False, host: str = "localhost"):
    if remote:
        data_path = "/data/"
        conn = psycopg2.connect(database="mobilitydb", user="docker", password="docker", host=host, port=5432)
    else:
        data_path = "./data/"
        conn = psycopg2.connect(database="mobilitydb", user="docker", password="docker", host=host, port=25432)
    print("ingest all scenic data")
    import_tables(conn, data_path)
    ingest_road(conn,data_path)

def ingest_road(conn, data_path):
    with open(data_path + 'road_network/polygon.json', 'r') as f:
        polygons = json.load(f)
    create_polygon_table(conn, polygons)

    with open(data_path + 'road_network/segment.json', 'r') as f:
        segments = json.load(f)
    create_segment_table(conn, segments)

    with open(data_path + 'road_network/laneSection.json', 'r') as f:
        laneSections = json.load(f)
    create_lanesection_table(conn, laneSections)

    with open(data_path + 'road_network/lane.json', 'r') as f:
        lanes = json.load(f)
    create_lane_table(conn, lanes)

    with open(data_path + 'road_network/lane_LaneSec.json', 'r') as f:
        lane_laneSec = json.load(f)
    create_lane_lanesec_table(conn, lane_laneSec)

    with open(data_path + 'road_network/laneGroup.json', 'r') as f:
        laneGroups = json.load(f)
    create_lanegroup_table(conn, laneGroups)

    with open(data_path + 'road_network/laneGroup_Lane.json', 'r') as f:
        laneGroup_lane = json.load(f)
    create_lanegroup_lane_table(conn, laneGroup_lane)

    with open(data_path + 'road_network/laneGroup_opposite.json', 'r') as f:
        opposite_lanegroup = json.load(f)
    create_opposite_lanegroup_table(conn, opposite_lanegroup)

    with open(data_path + 'road_network/road.json', 'r') as f:
        roads = json.load(f)
    create_road_table(conn, roads)

    with open(data_path + 'road_network/road_laneGroup.json', 'r') as f:
        road_laneGroups = json.load(f)
    create_road_lanegroup_table(conn, road_laneGroups)

    with open(data_path + 'road_network/road_roadSec.json', 'r') as f:
        road_roadSecs = json.load(f)
    create_road_roadsec_table(conn, road_roadSecs)

    with open(data_path + 'road_network/roadSection.json', 'r') as f:
        roadSections = json.load(f)
    create_roadsection_table(conn, roadSections)

    with open(data_path + 'road_network/roadSec_laneSec.json', 'r') as f:
        roadSec_laneSecs = json.load(f)
    create_roadsec_lanesec_table(conn, roadSec_laneSecs)

    with open(data_path + 'road_network/intersection.json', 'r') as f:
        intersections = json.load(f)
    create_intersection_table(conn, intersections)

    print("ingestion done")

def parse_args():
    parser=argparse.ArgumentParser(description="tell db the host address")
    parser.add_argument('--remote', action='store_true', help='remote db')
    parser.add_argument('--host', default='localhost', help='host address')
    args=parser.parse_args()
    return args

def main():
    args=parse_args()
    host=args.host
    remote=args.remote
    print("remote:", remote)
    ingest_data(remote=remote, host=host)
    

if __name__ == '__main__':
    main()