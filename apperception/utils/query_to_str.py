from typing import Union

from pypika.dialects import Query


def query_to_str(query: Union[str, Query]) -> str:
    if isinstance(query, str):
        return query
    return query.get_sql()
