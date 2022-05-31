from typing import TYPE_CHECKING

from .view import View

if TYPE_CHECKING:
    from ..query_type import QueryType


class TrajectoryView(View):
    object_id = "itemId"
    object_type = "objectType"
    color = "color"
    trajectory = "trajCentroids"
    traj = "trajCentroids"
    heading = "itemHeadings"
    table_name = "Item_General_Trajectory"
    table_type: "QueryType" = "TRAJ"

    def __init__(self):
        super().__init__(self.table_name, self.table_type)
        self.default = True
