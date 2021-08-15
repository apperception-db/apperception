# world-db: a database management system optimized for multi-video applications

World-db ingests video data from many perspectives and makes them queryable as a single multidimensional visual object. It incorporates new techniques for optimizing, executing, and storing multi-perspective video data. 

### How to Setup Apperception Repo

For ssh:
```
git clone git@github.com:apperception-db/apperception.git
cd apperception
```

For HTTPS:
```
git clone https://github.com/apperception-db/apperception.git
cd apperception
```
Then we setup the repo
chmod u+x ./setup.sh
chmod 733 ./setup.sh
./setup.sh
python3 object_tracker.py
```
## Apperception Demo Tryout without TASM
As TASM requires nividia-docker/nvidia-docker2(https://www.ibm.com/docs/en/maximo-vi/8.2.0?topic=planning-installing-docker-nvidia-docker2) during runtime, and a machine with an encode-capable GPU (https://developer.nvidia.com/video-encode-and-decode-gpu-support-matrix-new). To tryout Apperception features without TASM, run the following:
### Start Apperception Metadata Store MobilityDB(https://github.com/MobilityDB/MobilityDB)
```
docker volume create mobilitydb_data
docker run --name "mobilitydb" localhost -d -p 25432:25432 -v mobilitydb_data:/var/lib/postgresql mobilitydb/mobilitydb
```
We need to setup the mobilitydb with customized functions
```
cd pg_extender
psql -d mobilitydb -U docker
Enter "docker" as the default password
\i overlap.sql;
\q
```
# Try the demo.
In the docker:
`jupyter notebook`

The demo notebook first constructs the world. Then it queries for the trajectory of the cars that appeared once in an area of interests within some time interval.

To fully activate apperception in TASM:
```
docker-compose up
cd pg_extender
psql -h 172.19.0.3 -d mobilitydb -U docker
Enter "docker" as the default password
\i overlap.sql
\q
docker ps
```
After fetching the CONTAINER_ID of apperceptiontasm/tasm:latest, run
```
docker exec -it {CONTAINER_ID of apperceptiontasm/tasm:latest} /bin/bash
```
Now we are under TASM env
```
cd /apperception/
pip3 install -r requirements.txt
```
# Try the demo.
In the docker:  
`jupyter notebook --ip 172.19.0.2 --port 8890 --allow-root &`
Directly open the jupyter url
The demo notebook first constructs the world. Then it queries for the trajectory and videos of the cars that appeared once in an area of interests within some time interval.


