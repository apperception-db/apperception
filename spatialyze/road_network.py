from spatialyze.database import Database
from spatialyze.utils.ingest_road import ingest_location


class RoadNetwork:
    def __init__(self, location: "str", road_network_dir: "str"):
        self.location = location
        self.road_network_dir = road_network_dir

    def ingest(self, database: "Database"):
        ingest_location(database, self.road_network_dir, self.location)
