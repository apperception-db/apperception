from pypika.dialects import QueryBuilder, Query


def query_to_str(query: "str | QueryBuilder | Query") -> str:
    if isinstance(query, str):
        return query
    if isinstance(query, Query):
        return query._builder().get_sql()
    return query.get_sql()
