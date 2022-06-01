from typing import List

import cv2
import numpy as np


def get_video_roi(file_name: str, cam_video_file: str, rois: np.ndarray, times: List[int]):
    """
    Get the region of interest from the video, based on bounding box points in
    video coordinates.

    Args:
        file_name: String of file name to save video as
        rois: A list of bounding boxes
        time_intervals: A list of time intervals of which frames
    """

    rois = np.array(rois).T
    print(rois.shape)
    len_x, len_y = np.max(rois.T[2] - rois.T[0]), np.max(rois.T[3] - rois.T[1])
    # len_x, len_y  = np.max(rois.T[0][1] - rois.T[0][0]), np.max(rois.T[1][1] - rois.T[1][0])

    len_x = int(round(len_x))
    len_y = int(round(len_y))
    # print(len_x)
    # print(len_y)
    vid_writer = cv2.VideoWriter(
        file_name, cv2.VideoWriter_fourcc("m", "p", "4", "v"), 30, (len_x, len_y)
    )
    # print("rois")
    # print(rois)
    start_time = int(times[0])
    cap = cv2.VideoCapture(cam_video_file)
    frame_cnt = 0
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        if frame_cnt in times and ret:
            i = frame_cnt - start_time
            if i >= len(rois):
                print("incorrect length:", len(rois))
                break
            current_roi = rois[i]

            b_x, b_y, e_x, e_y = current_roi
            b_x, b_y = max(0, b_x), max(0, b_y)
            # e_x, e_y = current_roi[1]
            e_x, e_y = max(0, e_x), max(0, e_y)
            diff_y, diff_x = int(abs(e_y - b_y)), int(abs(e_x - b_x))
            pad_y = int((len_y - diff_y) // 2)
            pad_x = int((len_x - diff_x) // 2)

            # print("padding")
            # print(pad_y)
            # print(pad_x)
            roi_byte = frame[int(b_y) : int(e_y), int(b_x) : int(e_x), :]

            roi_byte = np.pad(
                roi_byte,
                pad_width=[
                    (pad_y, len_y - diff_y - pad_y),
                    (pad_x, len_x - diff_x - pad_x),
                    (0, 0),
                ],
            )
            frame = cv2.cvtColor(roi_byte, cv2.COLOR_RGB2BGR)

            vid_writer.write(roi_byte)
        frame_cnt += 1
        if not ret:
            break

    vid_writer.release()
