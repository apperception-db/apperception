import sys

if "./submodules" not in sys.path:
    sys.path.append("./submodules")

from pathlib import Path

FILE = Path(__file__).resolve()
SUBMODULE = FILE.parent.parent / "submodules"
YOLOV5_STRONGSORT_OSET_ROOT = SUBMODULE / "Yolov5_StrongSORT_OSNet"

if str(YOLOV5_STRONGSORT_OSET_ROOT) not in sys.path:
    sys.path.append(str(YOLOV5_STRONGSORT_OSET_ROOT))
if str(YOLOV5_STRONGSORT_OSET_ROOT / "yolov5") not in sys.path:
    sys.path.append(str(YOLOV5_STRONGSORT_OSET_ROOT / "yolov5"))
if str(YOLOV5_STRONGSORT_OSET_ROOT / "strong_sort") not in sys.path:
    sys.path.append(str(YOLOV5_STRONGSORT_OSET_ROOT / "strong_sort"))
if str(YOLOV5_STRONGSORT_OSET_ROOT / "strong_sort/deep/reid") not in sys.path:
    sys.path.append(str(YOLOV5_STRONGSORT_OSET_ROOT / "strong_sort/deep/reid"))
