from ..query_type import QueryType
from .view import View


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
