# Apperception: a database management system optimized for multi-video applications

Apperception ingests video data from many perspectives and makes them queryable as a single multidimensional visual object. It incorporates new techniques for optimizing, executing, and storing multi-perspective video data. 

## How to Setup Apperception Repo
### Install dependencies:
#### Debian based Linux
```sh
apt-get update && apt-get install -y postgresql python3-opencv
```
#### macOS
```sh
brew install postgresql
```
### Clone the Apperception repo
For ssh:
```sh
git clone git@github.com:apperception-db/apperception.git
cd apperception
```
For HTTPS:
```sh
git clone https://github.com/apperception-db/apperception.git
cd apperception
```
### Downloading Official YOLOv4 Pre-trained in the repo

Copy and paste yolov4.weights from your downloads folder into this repository. For the Demo, we use yolov4-tiny.weights,

If you want to use yolov4-tiny.weights, a smaller model that is faster at running detections but less accurate, download file here: https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights

Our object tracker uses YOLOv4 to make the object detections, which deep sort then uses to track. There exists an official pre-trained YOLOv4 object detector model that is able to detect 80 classes. For easy demo purposes we will use the pre-trained weights for our tracker. Download pre-trained yolov4.weights file: https://drive.google.com/open?id=1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT

Download the yolov4 model, unzip in the current repo
https://drive.google.com/file/d/1g5D0pU-PKoe7uTHI7cRjFlGRm-xRQ1ZL/view?usp=sharing

### Then we setup the repo
```sh
chmod u+x ./setup.sh
chmod 733 ./setup.sh
./setup.sh
```
## Apperception Demo Tryout without TASM
As TASM requires nividia-docker/nvidia-docker2(https://www.ibm.com/docs/en/maximo-vi/8.2.0?topic=planning-installing-docker-nvidia-docker2) during runtime, and a machine with an encode-capable GPU (https://developer.nvidia.com/video-encode-and-decode-gpu-support-matrix-new). To tryout Apperception features without TASM, run the following:
### Start Apperception Metadata Store MobilityDB(https://github.com/MobilityDB/MobilityDB)
```sh
docker volume create mobilitydb_data
docker run --name "mobilitydb" -d -p 25432:5432 -v mobilitydb_data:/var/lib/postgresql mobilitydb/mobilitydb
```
We need to setup the mobilitydb with customized functions
```sh
cd pg_extender
psql -h localhost -p 25432 -d mobilitydb -U docker
Enter "docker" as the default password
\i overlap.sql;
\q
```

### Try the demo.
In apperception repo:
`jupyter notebook` or `python3 -m notebook`

The demo notebook first constructs the world. Then it queries for the trajectory of the cars that appeared once in an area of interests within some time interval.

## To fully activate apperception in TASM:
```sh
docker-compose up
cd pg_extender
psql -h 172.19.0.3 -d mobilitydb -U docker
Enter "docker" as the default password
\i overlap.sql
\q
docker ps
```
After fetching the CONTAINER_ID of apperceptiontasm/tasm:latest, run
```sh
docker exec -it {CONTAINER_ID of apperceptiontasm/tasm:latest} /bin/bash
```
Now we are under TASM env
```sh
cd /apperception/
pip3 install -r requirements.txt
```
### Try the demo.
In the docker:  
`jupyter notebook --ip 172.19.0.2 --port 8890 --allow-root &`
Directly open the jupyter url
The demo notebook first constructs the world. Then it queries for the trajectory and videos of the cars that appeared once in an area of interests within some time interval.


