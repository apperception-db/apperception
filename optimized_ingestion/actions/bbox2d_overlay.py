import numpy as np
import cv2
import os

from ..stages.detection_2d.detection_2d import Detection2D, Metadatum
from ..stages.decode_frame.decode_frame import DecodeFrame
from ..payload import Payload


def bbox2d_overlay(payload: "Payload", base_dir: "str"):
    d2ds = payload[Detection2D]
    assert d2ds is not None

    images = payload[DecodeFrame]
    assert images is not None

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    videofile = os.path.join(base_dir, payload.video.videofile.split('/')[-1])
    print('dir', videofile)
    out = cv2.VideoWriter(videofile, fourcc, int(payload.video.fps), payload.video.dimension)

    for d2d, image in zip(d2ds, images):
        assert isinstance(d2d, Metadatum)
        assert isinstance(image, np.ndarray)

        detections, *_ = d2d
        for d in detections:
            l, t, w, h = d[:4]
            start = (int(l), int(t))
            end = (int(l + w), int(t + h))
            color = (255, 0, 0)
            thickness = 2
            image = cv2.rectangle(image, start, end, color, thickness)
        
        out.write(image)

    out.release()
    cv2.destroyAllWindows()
