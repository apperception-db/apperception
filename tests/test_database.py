import pytest
from apperception.database import database
from apperception.utils import import_tables

TABLE_NAMES = [
    "Cameras",
    "General_Bbox",
    "Item_General_Trajectory",
]


def test_reset():
    import_tables(database, './data')
    for t in TABLE_NAMES:
        assert not database.sql(f"select * from {t} limit 1").empty

    database.reset()
    for t in TABLE_NAMES:
        assert database.sql("select * from " + t).empty
