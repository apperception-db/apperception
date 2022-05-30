from apperception.data_types import QueryType
from apperception.data_types.views import (CameraView, LocationView,
                                           TrajectoryView, View)


class MetadataView(View):
    view_name = "metadata_view"
    view_type = QueryType.METADATA
    object_id = TrajectoryView.object_id
    object_type = TrajectoryView.object_type
    color = TrajectoryView.color
    trajectory = TrajectoryView.trajectory
    location = LocationView.location
    timestamp = LocationView.timestamp
    view_map = {
        object_id: TrajectoryView,
        object_type: TrajectoryView,
        color: TrajectoryView,
        trajectory: TrajectoryView,
        location: LocationView,
    }

    def __init__(self):
        super().__init__(self.view_name, self.view_type)
        self.default = True
        self.trajectory_view = TrajectoryView()
        self.location_view = LocationView()
        self.camera_view = CameraView()

    def map_view(self, column_key: str):
        if self.view_map[column_key] == TrajectoryView:
            return self.trajectory_view
        elif self.view_map[column_key] == LocationView:
            return self.location_view
        else:
            return self.camera_view

    def resolve_key(self, column_key: str):
        return (
            self.trajectory_view.resolve_key(column_key)
            or self.location_view.resolve_key(column_key)
            or self.camera_view.resolve_key(column_key)
            or column_key
        )
