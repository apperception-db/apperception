import ast
from dataclasses import dataclass
from datetime import datetime
from typing import List, Union


@dataclass
class SQLRepr:
    select: str
    from_: Union["SQLRepr", str]
    where: str
    join: str

    def to_str(self):
        # TODO
        return f"""
        select {self.select}
        from {self.from_}
        {'join'}
        """


@dataclass
class WorldSchema:
    objects: int
    camera: bool


@dataclass
class World:
    id: str
    parents: List["World"]
    done: bool
    timestamp: datetime
    materialized: bool
    schema: "WorldSchema"


@dataclass
class FilterWorld(World):
    predicate: ast.Expr

    def generate_sql(self):
        pass


@dataclass
class JoinWorld(World):
    predicate: ast.Expr

    def generate_sql(self):
        pass


@dataclass
class UnionWorld(World):
    pass

    def generate_sql(self):
        pass
