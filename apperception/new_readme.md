# New query engine

## Prerequisite
- Create a docker container for mobilitydb

```
docker volume create mobilitydb_data
docker run --name "mobilitydb" -d -p 25432:5432 -v mobilitydb_data:/var/lib/postgresql mobilitydb/mobilitydb
```

## Design

### Data Persistency
### Query Optimization

- Each world will have access to the list of operations done since the first world.
- Each item in the DB will have a row indicating the world id that creates it.
- When call operations on a world, we do not execute them right away. Instead, we collect these operations to be a sequence of operations.
- When we call get-data method (such as selectKey / get_trajectories) we construct an SQL - query based on the sequence of operations.
- When see predicates, we nest the query of the old world. 
- When see update, we concat the query of the old world.
- We can add cache on each node to accelerate.


