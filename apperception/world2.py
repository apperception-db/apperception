import ast
from dataclasses import dataclass
from datetime import datetime
from typing import List, Union
from uuid import uuid4


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


@dataclass(frozen=True)
class WorldSchema:
    objects: int
    camera: bool


@dataclass
class World:
    id: str
    parents: List["World"]
    schema: "WorldSchema"

    def filter(self, predicate) -> "FilterWorld":
        # TODO: schema should change if we have a predicate that involve camera
        # TODO: predicate should be translated from str??
        return FilterWorld(uuid4(), [self], self.schema, predicate)

    def join(self, *others: "World") -> "JoinWorld":
        pass

    def union(self, *others: "World") -> "UnionWorld":
        pass


@dataclass
class FilterWorld(World):
    predicate: ast.Expr

    def generate_sql(self):
        pass


@dataclass
class JoinWorld(World):
    predicate: ast.Expr
    schema_mapping: dict

    def generate_sql(self):
        pass


@dataclass
class UnionWorld(World):

    def generate_sql(self):
        pass
