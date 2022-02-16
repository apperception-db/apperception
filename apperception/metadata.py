from enum import Enum
class View:
    def __init__(self, view_name):
        self.view_name = view_name
        self.default = False
    
    def from_context(self, context):
        self.context = context
    
    def resolve_key(self, column_key):
        if column_key in self.__class__.__dict__:
            return self.__class__.__dict__[column_key]
        else:
            return None
    
    def contain(self, column_key):
        return column_key in self.__dict__.keys()
    
class TrajectoryView(View):
    object_id = "itemId"
    object_type = "objectType"
    color = "color"
    trajectory = "trajCentroids"
    table_name = "Test_Scenic_Item_General_Trajectory"
    def __init__(self):
        super().__init__(self.table_name)
        self.default = True
    
class LocationView(View):
    location = "trajBbox"
    timestamp = "timestamp"
    table_name = "Test_Scenic_General_Bbox"
    def __init__(self):
        super().__init__(self.table_name)
        self.default = True
        
class MetadataView(View):
    view_name = "metadata_view"
    object_id = TrajectoryView.object_id
    object_type = TrajectoryView.object_type
    color = TrajectoryView.color
    trajectory = TrajectoryView.trajectory
    location = LocationView.location
    timestamp = LocationView.timestamp
    view_map = {object_id:TrajectoryView,
                object_type:TrajectoryView,
                 color:TrajectoryView,
                 trajectory:TrajectoryView,
                 location:LocationView}
    def __init__(self):
        super().__init__(self.view_name)
        self.default = True
        self.trajectory_view = TrajectoryView()
        self.location_view = LocationView()

    def map_view(self, column_key):
        if self.view_map[column_key] == TrajectoryView:
            return self.trajectory_view
        else:
            return self.location_view
        
    def resolve_key(self, column_key):
        return self.trajectory_view.resolve_key(column_key) or self.location_view.resolve_key(column_key)
metadata_view = MetadataView()