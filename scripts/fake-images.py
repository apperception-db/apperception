import shutil
from os.path import join

DIR = "./data/scenic"

with open(join(DIR, "image_files.txt"), "r") as f:
    lines = f.readlines()

for line in lines:
    shutil.copyfile(join(DIR, "images/example.jpg"), join(DIR, "images", line))