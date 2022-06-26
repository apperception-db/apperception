from typing import Union

from pypika.dialects import QueryBuilder


def query_to_str(query: Union[str, QueryBuilder]) -> str:
    if isinstance(query, str):
        return query
    return query.get_sql()
