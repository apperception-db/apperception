import cv2
from bitarray import bitarray
from tqdm import tqdm
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple
import multiprocessing
from multiprocessing import Pool
import math

from .decode_frame import DecodeFrame

if TYPE_CHECKING:
    import numpy.typing as npt

    from ...payload import Payload


def decode(args: "Tuple[str, int, int]"):
    videofile, start, frames = args
    cap = cv2.VideoCapture(videofile)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    out: "List[npt.NDArray]" = []
    for _ in range(frames):
        ret, frame = cap.read()
        if not ret:
            break
        out.append(frame)
    cap.release()
    return out, start


class ParallelDecodeFrame(DecodeFrame):
    def _run(self, payload: "Payload") -> "Tuple[Optional[bitarray], Optional[Dict[str, list]]]":
        metadata: "List[npt.NDArray]" = []

        n_cpus = multiprocessing.cpu_count()
        n_frames = len(payload.video)
        frames_per_cpu = math.ceil(n_frames / n_cpus)
        with Pool(n_cpus) as pool:
            inputs = ((payload.video.videofile, i * frames_per_cpu, frames_per_cpu) for i in range(n_cpus))
            out = [*tqdm(pool.imap_unordered(decode, inputs), total=n_cpus)]
            for o, _ in sorted(out, key=lambda x: x[1]):
                metadata.extend(o)
        cv2.destroyAllWindows()

        assert len(metadata) == len(payload.video), (len(metadata), len(payload.video))

        return None, {self.classname(): metadata}
