from typing import Union, List, Tuple
import numpy as np
import cv2

def get_video_box(
    file_name: str,
    cam_video_file: str,
    rois: Union[List[Tuple[int, int, int, int]], np.ndarray],
    times: List[int],
):
    """
    Get the frames of interest from the video, while boxing in the object at interest
    with a box.

    Args:
        file_name: String of file name to save video as
        rois: A list of bounding boxes
        time_intervals: A list of time intervals of which frames
    """

    np_rois = np.array(rois).T
    print(np_rois.shape)

    cap = cv2.VideoCapture(cam_video_file)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vid_writer = cv2.VideoWriter(
        file_name, cv2.VideoWriter_fourcc("m", "p", "4", "v"), 30, (width, height)
    )

    start_time = int(times[0])
    frame_cnt = 0
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        if frame_cnt in times and ret:
            i = frame_cnt - start_time
            if i >= len(np_rois):
                print("incorrect length:", len(np_rois))
                break
            current_roi = np_rois[i]

            x1, y1, x2, y2 = current_roi
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 2)

            vid_writer.write(frame)
        frame_cnt += 1
        if not ret:
            break

    vid_writer.release()