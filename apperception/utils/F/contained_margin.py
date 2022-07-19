from __future__ import annotations

import ast
from typing import TYPE_CHECKING, List

from .fake_fn import fake_fn

if TYPE_CHECKING:
    from ..fn_to_sql import GenSqlVisitor


@fake_fn
def contained_margin(visitor: "GenSqlVisitor", args: List[ast.expr]):
    point, geoms, margin = args[:3]
    if len(args) == 4:
        point = point @ args[3]
    elif len(args) != 3:
        raise Exception("contained_margin accept either 3 or 4 arguments")

    return f"containedMargin({','.join(visitor, [point, geoms, margin])})"
