from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import os
import random
from typing import Any, Dict, List, Union


def show_images(
    base_dir: str,
    images: Union[List[str], Dict[str, Any]],
    sample: int = None,
    seed: int = 0
):
    _images: Union[List[str], Dict[str, Any]] = images
    if sample is not None and sample < len(images):
        print(f"Sampling {sample} out of {len(images)} images.")
        _images = [i for i in images]
        random.Random(seed).shuffle(_images)
        _images = _images[:sample]

    plt.figure(figsize=(60, 30))
    columns = 3

    for i, _image in enumerate(_images):
        print("image", _image)
        if isinstance(images, dict):
            for ii in images[_image]:
                print(ii)
        img = mpimg.imread(os.path.join(base_dir, _image))
        print("loaded")
        plt.subplot(len(_images) // columns + 1, columns, i + 1)
        plt.imshow(img)
