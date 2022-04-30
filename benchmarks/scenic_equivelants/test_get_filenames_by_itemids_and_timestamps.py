import sys
sys.path.insert(0, "./")

from apperception.new_world import *

from datetime import datetime, timezone

name = 'ScenicWorld' # world name
world = empty_world(name=name)


d = datetime(2018, 7, 23, 20, 28, 47, 604844, tzinfo=timezone.utc)
filenames = world.get_filenames_by_itemids_and_timestamps(["6dd2cbf4c24b4caeb625035869bca7b5"], [d])
print(filenames)
