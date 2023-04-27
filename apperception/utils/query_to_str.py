from pypika.dialects import Query, QueryBuilder


def query_to_str(query: "str | QueryBuilder | Query") -> str:
    if isinstance(query, str):
        return query
    assert not isinstance(query, Query)
    return query.get_sql()
