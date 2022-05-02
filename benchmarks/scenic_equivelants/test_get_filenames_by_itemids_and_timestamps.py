import sys
sys.path.insert(0, "./")

from apperception.new_world import *

from datetime import datetime, timezone

name = 'ScenicWorld' # world name
world = empty_world(name=name)

d = datetime(2018, 8, 1, 12, 54, 13, 912404, tzinfo=timezone.utc)
filenames = world.get_filenames_by_itemids_and_timestamps(["faf5a1c0630840339ff84fb0c8dac38a"], [d])
print(filenames)
