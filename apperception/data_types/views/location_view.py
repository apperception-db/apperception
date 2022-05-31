from typing import TYPE_CHECKING
from .view import View

if TYPE_CHECKING:
    from ..query_type import QueryType


class LocationView(View):
    location = "trajBbox"
    timestamp = "timestamp"
    table_name = "General_Bbox"
    table_type: "QueryType" = "BBOX"

    def __init__(self):
        super().__init__(self.table_name, self.table_type)
        self.default = True
