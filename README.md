# world-db: a database management system optimized for multi-video applications

World-db ingests video data from many perspectives and makes them queryable as a single multidimensional visual object. It incorporates new techniques for optimizing, executing, and storing multi-perspective video data. 

### How to Setuo Apperception Repo

For ssh:
```
git clone git@github.com:apperception-db/apperception.git
cd apperception
```

For HTTPS:
```
git clone https://github.com/apperception-db/apperception.git
cd apperception
chmod u+x ./setup.sh
chmod 733 ./setup.sh
./setup.sh
python3 object_tracker.py
```

### Start Apperception Env
```
docker volume create mobilitydb_data
docker-compose up -d
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
`jupyter notebook --ip 0.0.0.0 --port 8890 --allow-root &`

copy the jupyter notebook url
open up a browser, paste the url, and replace the hostname with 172.19.0.2 which is the static host for apperception docker container

The demo notebook first constructs the world. Then it queries for the trajectory and videos of the cars that appeared once in an area of interests within some time interval.


