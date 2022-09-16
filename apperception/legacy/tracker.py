from typing import Callable, Optional

import cv2
import numpy as np


class Tracker:
    def __init__(
        self, tracker_type="default", customized_tracker: "Optional[Callable[[], Tracker]]" = None
    ):
        """
        Constructs a Tracker.
        Args:
                tracker_type: indicator of whether using customized tracker
                customized_tracker: user specified tracker algorithm
        """
        self.tracker_type = tracker_type
        self.customized_tracker = customized_tracker

    def video_track(self, video_data, bboxes, first_frame):
        self.video_data = video_data
        if self.tracker_type == "default":
            self.tracker = SingleObjectTracker()
            return self.tracker.video_track(video_data, bboxes[0], first_frame)
        elif self.tracker_type == "multi":
            self.tracker = MultiObjectsTracker()
            print("boxes at tracker", bboxes)
            return self.tracker.video_track(video_data, bboxes, first_frame)
        elif self.customized_tracker is not None:
            self.tracker = self.customized_tracker()
            return self.tracker.video_track(video_data, bboxes, first_frame)
        raise Exception()

    def __iter__(self):
        return iter(self.tracker)

    def __next__(self):
        return next(self.tracker)


class SingleObjectTracker(Tracker):
    """
    OpenCV Single Object Tracker
    https://www.pyimagesearch.com/2018/07/30/opencv-object-tracking/
    """

    def __init__(self, tracker_type="CSRT"):
        """
        Constructs a Tracker.
        Args:
                tracker_type: type of the opencv tracker, default to be "CSRT"
        """
        self.tracker = cv2.TrackerCSRT_create()

    def video_track(self, video_data, bbox, first_frame):
        self.video_data = video_data
        if self.tracker.init(first_frame, bbox):
            return iter(self)
        else:
            return None

    def __iter__(self):
        self.video_iter = iter(self.video_data)
        self.framect = 0
        return self

    def __next__(self):
        frame = next(self.video_iter)
        self.framect += 1
        ok, bbox = self.tracker.update(frame)
        if ok:
            p1 = [int(bbox[0]), int(bbox[1])]
            p2 = [int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])]
            cv2.rectangle(frame, p1, p2, (255, 255, 255), 2, 2)
        else:
            p1 = [0, 0]
            p2 = [0, 0]
            # Tracking failure
            cv2.putText(
                frame,
                "Tracking failure detected",
                (100, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 0, 255),
                2,
            )

        # Return the new bounding box and frameidx
        return frame, [[p1, p2]], self.framect


class MultiObjectsTracker(Tracker):
    """
    OpenCV Multi Object Tracker
    https://www.pyimagesearch.com/2018/08/06/tracking-multiple-objects-with-opencv/
    """

    def __init__(self, tracker_type="Multi"):
        """
        Constructs a Tracker.
        Args:
                tracker_type: type of the opencv tracker, default to be "CSRT"
        """
        self.trackers = []

    def video_track(self, video_data, bboxes, first_frame):
        # print(bboxes)
        self.video_data = video_data
        for bbox in bboxes:
            tracker = cv2.TrackerCSRT_create()
            tracker.init(first_frame, bbox)
            self.trackers.append(tracker)
        return iter(self)

    def __iter__(self):
        self.video_iter = iter(self.video_data)
        self.framect = 0
        return self

    def __next__(self):
        frame = next(self.video_iter)
        self.framect += 1
        tracker_boxes = np.zeros((len(self.trackers), 2, 2))
        for i in range(len(self.trackers)):
            current_tracker = self.trackers[i]
            ok, bbox = current_tracker.update(frame)
            if ok:
                p1 = [int(bbox[0]), int(bbox[1])]
                p2 = [int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])]
                tracker_boxes[i] = np.array([p1, p2])
                # tracker_boxes.append([p1,p2])
                cv2.rectangle(frame, tuple(p1), tuple(p2), (255, 255, 255), 2, 2)
            else:
                # Tracking failure
                cv2.putText(
                    frame,
                    "Tracking failure detected, Tracker %d" % i,
                    (100, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 0, 255),
                    2,
                )

        return frame, tracker_boxes, self.framect
