from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import os
import random
from typing import List

def show_images(base_dir: str, images: List[str], sample: int = None):
    if sample is not None and sample < len(images):
        print(f"Sampling {sample} out of {len(images)} images.")
        images = [i for i in images]
        random.shuffle(images)
        images = images[:sample]
    
    plt.figure(figsize=(60,30))
    columns = 3

    for i, image in enumerate(images):
        print("image", image)
        img = mpimg.imread(os.path.join(base_dir, image))
        print("loaded")
        plt.subplot(len(images) // columns + 1, columns, i + 1)
        plt.imshow(img)