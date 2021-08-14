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
```
TO setup mobilitydb
```
cd pg_extender
psql -h 172.19.0.3 -d mobilitydb -U docker
Enter "docker" as the default password
\i overlap.sql;
\q
```
To fully activate apperception in TASM:
```
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

The demo notebook first constructs the world. Then it queries for the trajectory and videos of the cars that appeared once in an area of interests within some time interval.


