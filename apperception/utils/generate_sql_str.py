from typing import Dict, List, Optional, Tuple, Union

from apperception.data_types import SQL


def generate_sql_str(
    sql: "SQL",
    tables: Optional[Dict[str, str]] = None,
    schemas: Optional[Dict[str, List[str]]] = None,
):
    if tables is None:
        tables = {}
    if schemas is None:
        schemas = {}


def annotate_schema(sql: "SQL") -> SQL:
    if sql.schema is not None:
        return sql

    _table, _schema = annotate_table(sql.table)
    num_table = 0
    num_camera = 0
    for arg in sql.where.args.args:
        identifier = arg.arg
        if identifier in ["c", "cam", "camera"]:
            if num_camera > 0:
                raise Exception("Only allow one camera parameter")
            num_camera = 1
        else:
            num_table += 1
    if num_table < _schema[0]:
        num_table = _schema[0]

    if num_camera < _schema[1]:
        num_table = _schema[1]

    return SQL(sql.uuid, _table, sql.where, (num_table, num_camera))


def annotate_table(table: Union[str, SQL]) -> "Tuple[Union[str, SQL], Tuple[int, int]]":
    if isinstance(table, str):
        return table, (1, 0)

    _table = annotate_schema(table)
    if _table.schema is None:
        raise Exception("table's schema should be annotated")

    return _table, _table.schema
