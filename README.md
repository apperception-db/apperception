# world-db: a database management system optimized for multi-video applications

World-db ingests video data from many perspectives and makes them queryable as a single multidimensional visual object. It incorporates new techniques for optimizing, executing, and storing multi-perspective video data. 

### How to Setuo Apperception Repo
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
docker-compose up
```
Now we are under TASM env
```
cd /apperception/
```
# Try the demo.
In the docker:  
`jupyter notebook --ip 0.0.0.0 --port 8890 --allow-root &`

On the local machine:  
`ssh -L 8890:127.0.0.1:8890 <user>@<host>`

The demo notebook first constructs the world. Then it queries for the trajectory and videos of the cars that appeared once in an area of interests within some time interval.


