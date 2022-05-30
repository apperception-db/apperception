from apperception.data_types import QueryType
from apperception.data_types.views import View


class TrajectoryView(View):
    object_id = "itemId"
    object_type = "objectType"
    color = "color"
    trajectory = "trajCentroids"
    traj = "trajCentroids"
    heading = "itemHeadings"
    table_name = "Item_General_Trajectory"
    table_type = QueryType.TRAJ

    def __init__(self):
        super().__init__(self.table_name, self.table_type)
        self.default = True
