from apperception.data_types import QueryType

class View:
    def __init__(self, view_name: str, view_type: QueryType):
        self.view_name: str = view_name
        self.default: bool = False
        self.view_type: QueryType = view_type

    def from_context(self, context):
        self.context = context

    def resolve_key(self, column_key):
        if column_key in self.__class__.__dict__:
            return self.__class__.__dict__[column_key]
        else:
            return None

    def contain(self, column_key):
        return column_key in self.__dict__.keys()


class CameraView(View):
    camera_id = "cameraId"
    frame_id = "frameId"
    frame_num = "frameNum"
    file_name = "fileName"
    camera_translation = "cameraTranslation"
    camera_rotation = "cameraRotation"
    camera_intrinsic = "cameraIntrinsic"
    ego_translation = "egoTranslation"
    ego_rotation = "egoRotation"
    timestamp = "timestamp"
    camera_heading = "cameraHeading"
    ego_heading = "egoHeading"
    table_name = "Cameras"
    table_type = QueryType.CAM

    def __init__(self):
        super().__init__(self.table_name, self.table_type)
        self.default = True


class TrajectoryView(View):
    object_id = "itemId"
    object_type = "objectType"
    color = "color"
    trajectory = "trajCentroids"
    traj = "trajCentroids"
    table_name = "Item_General_Trajectory"
    table_type = QueryType.TRAJ

    def __init__(self):
        super().__init__(self.table_name, self.table_type)
        self.default = True


class LocationView(View):
    location = "trajBbox"
    timestamp = "timestamp"
    table_name = "General_Bbox"
    table_type = QueryType.BBOX

    def __init__(self):
        super().__init__(self.table_name, self.table_type)
        self.default = True


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

    def map_view(self, column_key):
        if self.view_map[column_key] == TrajectoryView:
            return self.trajectory_view
        elif self.view_map[column_key] == LocationView:
            return self.location_view
        else:
            return self.camera_view

    def resolve_key(self, column_key):
        return self.trajectory_view.resolve_key(column_key) or self.location_view.resolve_key(column_key) or self.camera_view.reseolve_key(column_key)


metadata_view = MetadataView()
