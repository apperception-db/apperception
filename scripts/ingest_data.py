# %%
import os
from apperception.database import Database, database
from apperception.utils import import_tables, ingest_road

# %%
EXPERIMENT = False

# %%
suffix = "_experiment" if EXPERIMENT else ""

# %%
if 'NUSCENES_PROCESSED_ROAD' in os.environ:
    road_data = os.environ['NUSCENES_PROCESSED_ROAD'] + "/boston-old"
else:
    road_data = '/data/processed/road-network/boston-seaport'

# %%
road_data

# %%
ingest_road(database, road_data)

# %%
from apperception.world import empty_world
import os
import pickle

# %%
world = empty_world("w")

# %%
# del os.environ['NUSCENES_PROCESSED_DATA']

# %%
if 'NUSCENES_PROCESSED_DATA' in os.environ:
    base_dir = os.environ['NUSCENES_PROCESSED_DATA']
else:
    base_dir = '/data/processed/full-dataset/mini/'
#     base_dir = '/work/apperception/data/nuScenes/full-dataset-v1.0/Trainval'
base_dir

# %%
# Or uncomment this cell
with open(os.path.join(base_dir, f'sample_data{suffix}.pkl'), "rb") as f:
    df_sample_data = pickle.loads(f.read())
with open(os.path.join(base_dir, f'annotation{suffix}.pkl'), "rb") as f:
    df_annotation = pickle.loads(f.read())

# %%
scenes = df_sample_data['scene_name'].drop_duplicates().values.tolist()

# %%
str(df_sample_data['ego_translation'][0])

# %%
from apperception.utils import df_to_camera_config
from apperception.data_types import Camera
database.reset()
for scene in scenes:
    print(scene, scenes[-1])
    config = df_to_camera_config(scene, df_sample_data)
    camera = Camera(config=config, id=scene)
    world <<= (camera, df_annotation)

# %%
world.get_traj_key()

# %%
from apperception.utils import export_tables

# %%
export_tables(database.connection, "../data/scenic/database/")

# %%



