import ast
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union


# @dataclass
# class SQL:
#     uuid: "str"
#     table: "Union[SQL, str]"
#     where: "ast.Lambda"
#     schema: "Optional[Tuple[int, int]]" = None


@dataclass
class FilterNode:
    # TODO: schema never change so we do not need to store schema here
    uuid: "str"
    table: "Union[SQL, str]"
    predicate: "ast.Lambda"


@dataclass
class JoinNode:
    # TODO: schema change, we need to store old schema, and new
    # old schema for each table.
    # new combined schema
    uuid: "str"
    others: "List[Union[SQL, str]]"


SQL = Union[FilterNode, JoinNode]

# @dataclass
# class Join:
#     table: "Union[SQL, str]"
#     on: "Union[ast.Lambda, List[str]]"
