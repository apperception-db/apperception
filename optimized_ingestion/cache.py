
import os
import pickle
from typing import TYPE_CHECKING, Callable, TypeVar

from .stages.stage import Stage

if TYPE_CHECKING:
    from .payload import Payload


_CACHE_STATUS = {
    "disable": False
}


def get_cache_filename(videofile: str, stage_name: str):
    return f'./.pipeline_cache/{stage_name}-{videofile}.pickle'


S = TypeVar('S')
T = TypeVar('T')
def cache(fn: "Callable[[S, Payload], T]") -> "Callable[[S, Payload], T]":
    def _fn(stage: "S", payload: "Payload"):
        if _CACHE_STATUS['disable']:
            return fn(stage, payload)

        videofile = payload.video.videofile.split("/")[-1]
        assert isinstance(stage, Stage)
        cache_filename = get_cache_filename(videofile, stage.classname())

        if os.path.exists(cache_filename):
            with open(cache_filename, "rb") as f:
                return pickle.load(f)
        else:
            result = fn(stage, payload)
            with open(cache_filename, "wb") as f:
                pickle.dump(result, f)
            return result
    
    return _fn


def disable_cache(disable: bool = True):
    _CACHE_STATUS['disable'] = disable