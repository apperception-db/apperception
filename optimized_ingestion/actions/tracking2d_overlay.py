import os

import cv2
import numpy as np

from ..payload import Payload
from ..stages.decode_frame.decode_frame import DecodeFrame
from ..stages.tracking_2d.tracking_2d import Tracking2D


def tracking2d_overlay(payload: "Payload", base_dir: "str"):
    t2ds = payload[Tracking2D]
    assert t2ds is not None

    images = payload[DecodeFrame]
    assert images is not None

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    videofile = os.path.join(base_dir, payload.video.videofile.split('/')[-1])
    # print('dir', videofile)
    out = cv2.VideoWriter(videofile, fourcc, int(payload.video.fps), payload.video.dimension)

    for t2d, image in zip(t2ds, images):
        assert isinstance(t2d, dict)
        assert isinstance(image, np.ndarray)

        for oid, det in t2d.items():
            oid = int(oid)
            l, t, w, h = det.bbox_left, det.bbox_top, det.bbox_w, det.bbox_h
            start = (int(l), int(t))
            end = (int(l + w), int(t + h))

            ccode = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"][oid % 10]
            ccode = ccode[1:]
            color = (int(ccode[4:], 16), int(ccode[2:4], 16), int(ccode[:2], 16))

            thickness = 2
            image = cv2.rectangle(image, start, end, color, thickness)

            image = cv2.putText(image, f'{det.object_type} {oid}', start, cv2.FONT_HERSHEY_SIMPLEX, 1, color, thickness, cv2.LINE_AA)
        out.write(image)

    out.release()
    cv2.destroyAllWindows()
