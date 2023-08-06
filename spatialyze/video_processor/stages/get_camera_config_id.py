from ..payload import Payload
from .stage import Stage


class GetCameraConfigId(Stage["str"]):
    def _run(self, payload: "Payload"):
        frame_ids = [f.frame_id for f in payload.video.interpolated_frames]
        assert all(fid is not None for fid in frame_ids), frame_ids
        return None, {self.classname(): frame_ids}
