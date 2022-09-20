# Optimized Ingestion

## Clone dependencies
```sh
git submodule --init --recursive
```

## Setup
```sh
poetry shell
poetry install
# OR install all the packages in ./pyproject.toml

# Then
pip install -r ./submodules/Yolov5_StrongSORT_OSNet/requirements.txt
```

## Data
1. Download NuScenes' mini dataset
2. Download generated videos from `/work/apperception/data/nuScenes/full-dataset-v1.0/Mini/videos` and place it inside `v1.0-mini`

## Run
```sh
# set port for mobilitydb
export AP_PORT="..."

# set directory for nuscene data (use directory to v1.0-mini)
export NUSCENE_DATA="..."

# run
python -m optimized_ingestion
```