# world-db: a database management system optimized for multi-video applications

World-db ingests video data from many perspectives and makes them queryable as a single multidimensional visual object. It incorporates new techniques for optimizing, executing, and storing multi-perspective video data. 

### How to start PostgreSQL Server

For Mac users:
```
brew services start postgresql
```

### Docker Setup with MobilityDB

Install Anaconda:
```
wget https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh
bash Anaconda3-2020.11-Linux-x86_64.sh
```

Create conda environment:
```
conda create -n py36 python=3.6
```

To enter environment:
```
source activate py36
```

Download dependencies
```
apt update
apt install libgl1-mesa-glx
pip install numpy
pip install Pillow
pip install matplotlib
pip install opencv-python-headless
pip install ffmpeg-python
pip install ffprobe-python
pip install psycopg2
```

Install .sql files on PSQL
```
\i pg_extender/boxtraj_to_centraj.sql
\i pg_extender/overlap.sql
\i pg_extender/stbox_contain.sql
```


