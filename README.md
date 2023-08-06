# Spatialyze: A Geospatial Video Analytic System with Spatial-Aware Optimizations

![Tests and Type Checks](https://github.com/apperception-db/spatialyze/actions/workflows/test-and-check.yml/badge.svg?branch=dev)

## Absract
Videos that are shot using commodity hardware such as phones
and surveillance cameras record various metadata such as time and
location. We encounter such geospatial videos on a daily basis and
such videos have been growing in volume significantly. Yet, we
do not have data management systems that allow users to interact
with such data effectively.

In this paper, we describe Spatialyze, a new framework for end-
to-end querying of geospatial videos. Spatialyze comes with a
domain-specific language where users can construct geospatial
video analytic workflows using a 3-step, declarative, build-filter-
observe paradigm. Internally, Spatialyze leverages the declarative
nature of such workflows, the temporal-spatial metadata stored
with videos, and physical behavior of real-world objects to optimize
the execution of workflows. Our results using real-world videos
and workflows show that Spatialyze can reduce execution time by
up to 5.3x, while maintaining up to 97.1% accuracy compared to
unoptimized execution.

## Requirement
```
python >= 3.10
```

## How to Setup Spatialyze Repo
### Install dependencies:
#### Debian based Linux
```sh
apt-get update && apt-get install -y python3-opencv
```
### Clone the Spatialyze repo
For ssh:
```sh
git clone -b optimized_ingestion --recurse-submodules git@github.com:apperception-db/spatialyze.git
cd spatialyze
```

### We use Conda/Mamba to manage our python environment
Install Mamba: https://mamba.readthedocs.io/en/latest/installation.html
or install Conda: https://docs.conda.io/en/latest/miniconda.html

### Setup Environment and Dependencies
```sh
# clone submodules
git submodule update --init --recursive

# setup virtual environment
# with conda
conda env create -f environment.yml
conda activate spatialyze
# OR with mamba
mamba env create -f environment.yml
mamba activate spatialyze

# install python dependencies
poetry install
pip install lap  # a bug in lap/poetry/conda that lap needs to be installed using pip.
```

## Spatialyze Demo
### Start Spatialyze Geospatial Metadata Store [MobilityDB](https://github.com/MobilityDB/MobilityDB)
```sh
docker volume create spatialyze-gs-store-data
docker run --name "spatialyze-gs-store" -d -p 25432:5432 -v spatialyze-gs-store-data:/var/lib/postgresql mobilitydb/mobilitydb
```
We need to setup the mobilitydb with customized functions
```sh
docker exec -it spatialyze-gs-store rm -rf /pg_extender
docker cp pg_extender spatialyze-gs-store:/pg_extender
docker exec -it -w /pg_extender spatialyze-gs-store python3 install.py
```
To run MobilityDB every system restart
```sh
docker update --restart unless-stopped spatialyze-gs-store
```

### Try the demo.
In spatialyze repo:
`jupyter notebook` or `python3 -m notebook`

The demo notebook first constructs the world. Then it queries for the trajectory of the cars that appeared once in an area of interests within some time interval.
