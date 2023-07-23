import psycopg2.sql


def create_sql(sql: "psycopg2.sql.Composable | str"):
    if isinstance(sql, str):
        return psycopg2.sql.SQL(sql)
    return sql
