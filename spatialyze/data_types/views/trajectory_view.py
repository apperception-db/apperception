from .view import View


class TrajectoryView(View):
    object_id = "itemId"
    object_type = "objectType"
    color = "color"
    trajectory = "trajCentroids"
    traj = "trajCentroids"
    heading = "itemHeadings"
    table_name = "Item_General_Trajectory"

    def __init__(self):
        super().__init__(self.table_name)
        self.default = True
