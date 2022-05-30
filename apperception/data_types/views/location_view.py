from apperception.data_types import QueryType
from apperception.data_types.views import View


class LocationView(View):
    location = "trajBbox"
    timestamp = "timestamp"
    table_name = "General_Bbox"
    table_type = QueryType.BBOX

    def __init__(self):
        super().__init__(self.table_name, self.table_type)
        self.default = True
