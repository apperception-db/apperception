from ...cache import cache
from ..detection_2d.detection_2d import Detection2D
from ..stage import Stage

from .construct_segment_trajectory import calibrate
from .utils import trajectory_3d


SegmentTrajectoryMetadatum = List[SegmentTrajectoryPoint]


class SegmentTrajectory(Stage[SegmentTrajectoryMetadatum]):
    @cache
    def _run(self, payload: "Payload"):
        if Detection2D.get(payload) is None:
            raise Exception()

        metadata = []
        obj_trajectories = construct_trajectory(payload)
        for trajectory_3d, detection_infos in obj_trajectories:
            metadata.append(calibrate(trajectory_3d, detection_infos))
        return None, {self.classname(): metadata}


def construct_trajectory(source: "Payload"):
    obj_3d_trajectories = {}
    for frame in source:
        for obj_id, obj_trajectory in frame.objects.items():
            if obj_id not in obj_3d_trajectories:
                obj_3d_trajectories[obj_id] = (
                    obj_trajectory[0], get_detection_infos(obj_trajectory[0]))
    return obj_3d_trajectories.values()
