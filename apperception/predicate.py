from typing import (Any, Callable, Dict, Generic, List, Literal, Optional, Set,
                    Tuple, TypeVar)

BinOp = Literal["add", "sub", "mul", "div", "matmul"]
BoolOp = Literal["and", "or"]
CompOp = Literal["eq", "ne", "gt", "ge", "lt", "le"]
UnaryOp = Literal["invert", "neg"]


class PredicateNode:
    def __init__(self, *args, **kwargs):
        anns = self.__annotations__.keys()
        if len(args) + len(kwargs) != len(anns):
            raise Exception(
                f"Mismatch number of arguments: expecting {len(anns)}, received {len(args)} args and {len(kwargs)} kwargs"
            )

        for k in kwargs:
            if k not in anns:
                raise Exception(f"{self.__class__.__name__} does not have attribute {k}")

        arg = iter(args)
        for k in anns:
            if k in kwargs:
                setattr(self, k, kwargs[k])
            else:
                setattr(self, k, next(arg))

    def __add__(self, other):
        other = wrap_literal(other)
        return BinOpNode(self, "add", other)

    def __radd__(self, other):
        other = wrap_literal(other)
        return BinOpNode(other, "add", self)

    def __sub__(self, other):
        other = wrap_literal(other)
        return BinOpNode(self, "sub", other)

    def __rsub__(self, other):
        other = wrap_literal(other)
        return BinOpNode(other, "sub", self)

    def __mul__(self, other):
        other = wrap_literal(other)
        return BinOpNode(self, "mul", other)

    def __rmul__(self, other):
        other = wrap_literal(other)
        return BinOpNode(other, "mul", self)

    def __truediv__(self, other):
        other = wrap_literal(other)
        return BinOpNode(self, "div", other)

    def __rtruediv__(self, other):
        other = wrap_literal(other)
        return BinOpNode(other, "div", self)

    def __matmul__(self, other):
        other = wrap_literal(other)
        return BinOpNode(self, "matmul", other)

    def __rmatmul__(self, other):
        other = wrap_literal(other)
        return BinOpNode(other, "matmul", self)

    def __and__(self, other):
        other = wrap_literal(other)
        return BoolOpNode("and", [self, other])

    def __rand__(self, other):
        other = wrap_literal(other)
        return BoolOpNode("and", [other, self])

    def __or__(self, other):
        other = wrap_literal(other)
        return BoolOpNode("or", [self, other])

    def __ror__(self, other):
        other = wrap_literal(other)
        return BoolOpNode("or", [other, self])

    def __eq__(self, other):
        other = wrap_literal(other)
        return CompOpNode(self, "eq", other)

    def __req__(self, other):
        other = wrap_literal(other)
        return CompOpNode(other, "eq", self)

    def __ne__(self, other):
        other = wrap_literal(other)
        return CompOpNode(self, "ne", other)

    def __rne__(self, other):
        other = wrap_literal(other)
        return CompOpNode(other, "ne", self)

    def __ge__(self, other):
        other = wrap_literal(other)
        return CompOpNode(self, "ge", other)

    def __rge__(self, other):
        other = wrap_literal(other)
        return CompOpNode(other, "ge", self)

    def __gt__(self, other):
        other = wrap_literal(other)
        return CompOpNode(self, "gt", other)

    def __rgt__(self, other):
        other = wrap_literal(other)
        return CompOpNode(other, "gt", self)

    def __le__(self, other):
        other = wrap_literal(other)
        return CompOpNode(self, "le", other)

    def __rle__(self, other):
        other = wrap_literal(other)
        return CompOpNode(other, "le", self)

    def __lt__(self, other):
        other = wrap_literal(other)
        return CompOpNode(self, "lt", other)

    def __rlt__(self, other):
        other = wrap_literal(other)
        return CompOpNode(other, "lt", self)

    def __invert__(self):
        return UnaryOpNode("invert", self)

    def __neg__(self):
        return UnaryOpNode("neg", self)

    def __repr__(self):
        return f"{self.__class__.__name__}({', '.join(f'{k}={getattr(self, k).__repr__()}' for k in self.__annotations__)})"


class ArrayNode(PredicateNode):
    exprs: List["PredicateNode"]


def wrap_literal(x: Any) -> "PredicateNode":
    if isinstance(x, list):
        return arr(*x)
    if not isinstance(x, PredicateNode):
        return LiteralNode(x, True)
    return x


def arr(*exprs: "PredicateNode"):
    return ArrayNode([*map(wrap_literal, exprs)])


class CompOpNode(PredicateNode):
    left: "PredicateNode"
    op: CompOp
    right: "PredicateNode"


class BinOpNode(PredicateNode):
    left: "PredicateNode"
    op: BinOp
    right: "PredicateNode"


class BoolOpNode(PredicateNode):
    op: BoolOp
    exprs: List["PredicateNode"]


class UnaryOpNode(PredicateNode):
    op: UnaryOp
    expr: "PredicateNode"


class LiteralNode(PredicateNode):
    value: Any
    python: bool


def lit(value: Any, python: bool = True):
    return LiteralNode(value, python)


class TableNode(PredicateNode):
    index: Optional[int]

    def __getattr__(self, name: str) -> "TableAttrNode":
        return TableAttrNode(name, self, False)

    def __repr__(self):
        if self.index is None:
            return self.__class__.__name__
        return f"{self.__class__.__name__}[{self.index}]"


class ObjectTableNode(TableNode):
    index: int

    def __init__(self, index: int):
        self.index = index
        self.traj = TableAttrNode("trajCentroids", self, True)
        self.trans = TableAttrNode("translations", self, True)
        self.id = TableAttrNode("itemId", self, True)
        self.type = TableAttrNode("objectType", self, True)


class CameraTableNode(TableNode):
    def __init__(self, index: Optional[int] = None):
        self.index = index
        self.time = TableAttrNode("timestamp", self, True)
        self.ego = TableAttrNode("egoTranslation", self, True)
        self.cam = TableAttrNode("cameraTranslation", self, True)
        self.camAbs = TableAttrNode("cameraTranslationAbs", self, True)


class TableAttrNode(PredicateNode):
    name: str
    table: "TableNode"
    shorten: bool


class ObjectTables:
    def __getitem__(self, index: int) -> "ObjectTableNode":
        if not isinstance(index, int):
            raise Exception("index must be an integer, instead received: ", type(index))
        return ObjectTableNode(index)


class CameraTables:
    def __getitem__(self, index: int) -> "CameraTableNode":
        if not isinstance(index, int):
            raise Exception("index must be an integer, instead received: ", type(index))
        return CameraTableNode(index)


objects = ObjectTables()
cameras = CameraTables()
camera = CameraTableNode()


Fn = Callable[["GenSqlVisitor", List["PredicateNode"]], str]


class CallNode(PredicateNode):
    _fn: Tuple["Fn"]
    params: List["PredicateNode"]

    def __init__(self, fn: "Fn", params: List["PredicateNode"]):
        self._fn = (fn,)
        self.params = params

    @property
    def fn(self) -> "Fn":
        return self._fn[0]


def call_node(fn: "Fn"):
    def call_node_factory(*args: "PredicateNode") -> "CallNode":
        return CallNode(
            fn, [arg if isinstance(arg, PredicateNode) else LiteralNode(arg, True) for arg in args]
        )

    return call_node_factory


class CastNode(PredicateNode):
    to: str
    expr: "PredicateNode"


T = TypeVar("T")


class Visitor(Generic[T]):
    def __call__(self, node: "PredicateNode") -> T:
        attr = f"visit_{node.__class__.__name__}"
        if not hasattr(self, attr):
            raise Exception("Unknown node type:", node.__class__.__name__)
        return getattr(self, attr)(node)

    def visit_ArrayNode(self, node: "ArrayNode") -> Any:
        for e in node.exprs:
            self(e)

    def visit_CompOpNode(self, node: "CompOpNode") -> Any:
        self(node.left)
        self(node.right)

    def visit_BinOpNode(self, node: "BinOpNode") -> Any:
        self(node.left)
        self(node.right)

    def visit_BoolOpNode(self, node: "BoolOpNode") -> Any:
        for e in node.exprs:
            self(e)

    def visit_UnaryOpNode(self, node: "UnaryOpNode") -> Any:
        self(node.expr)

    def visit_LiteralNode(self, node: "LiteralNode") -> Any:
        ...

    def visit_TableAttrNode(self, node: "TableAttrNode") -> Any:
        self(node.table)

    def visit_CallNode(self, node: "CallNode") -> Any:
        for p in node.params:
            self(p)

    def visit_TableNode(self, node: "TableNode") -> Any:
        ...

    def visit_ObjectTableNode(self, node: "ObjectTableNode") -> Any:
        ...

    def visit_CameraTableNode(self, node: "CameraTableNode") -> Any:
        ...

    def visit_CastNode(self, node: "CastNode") -> Any:
        self(node.expr)


class BaseTransformer(Visitor[PredicateNode]):
    def visit_ArrayNode(self, node: "ArrayNode"):
        return ArrayNode([self(e) for e in node.exprs])

    def visit_CompOpNode(self, node: "CompOpNode"):
        return CompOpNode(self(node.left), node.op, self(node.right))

    def visit_BinOpNode(self, node: "BinOpNode"):
        return BinOpNode(self(node.left), node.op, self(node.right))

    def visit_BoolOpNode(self, node: "BoolOpNode"):
        return BoolOpNode(node.op, [self(e) for e in node.exprs])

    def visit_UnaryOpNode(self, node: "UnaryOpNode"):
        return UnaryOpNode(node.op, self(node.expr))

    def visit_LiteralNode(self, node: "LiteralNode"):
        return node

    def visit_TableAttrNode(self, node: "TableAttrNode"):
        return TableAttrNode(node.name, self(node.table), node.shorten)

    def visit_CallNode(self, node: "CallNode"):
        return CallNode(node.fn, [self(p) for p in node.params])

    def visit_TableNode(self, node: "TableNode"):
        return node

    def visit_ObjectTableNode(self, node: "ObjectTableNode"):
        return node

    def visit_CameraTableNode(self, node: "CameraTableNode"):
        return node

    def visit_CastNode(self, node: "CastNode"):
        return CastNode(node.to, self(node.expr))


class ExpandBoolOpTransformer(BaseTransformer):
    def __call__(self, node: "PredicateNode"):
        if isinstance(node, BoolOpNode):
            exprs: List["PredicateNode"] = []
            for expr in node.exprs:
                e = self(expr)
                if isinstance(e, BoolOpNode) and e.op == node.op:
                    exprs.extend(e.exprs)
                else:
                    exprs.append(e)
            return BoolOpNode(node.op, exprs)
        return super().__call__(node)


class FindAllTablesVisitor(Visitor[Tuple[Set[int], bool]]):
    tables: Set[int]
    camera: bool

    def __init__(self):
        self.tables = set()
        self.camera = False

    def __call__(self, node: "PredicateNode"):
        super().__call__(node)
        return self.tables, self.camera

    def visit_ObjectTableNode(self, node: "ObjectTableNode"):
        self.tables.add(node.index)

    def visit_CameraTableNode(self, node: "CameraTableNode"):
        self.camera = True


class MapTablesTransformer(BaseTransformer):
    mapping: Dict[int, int]

    def __init__(self, mapping: Dict[int, int]):
        self.mapping = mapping

    def visit_ObjectTableNode(self, node: "ObjectTableNode"):
        if node.index in self.mapping:
            return objects[self.mapping[node.index]]
        return node


BIN_OP: Dict[BinOp, str] = {
    "add": "+",
    "sub": "-",
    "mul": "*",
    "div": "/",
}

BOOL_OP: Dict[BoolOp, str] = {
    "and": " AND ",
    "or": " OR ",
}

COMP_OP: Dict[CompOp, str] = {
    "eq": "=",
    "ne": "<>",
    "ge": ">=",
    "gt": ">",
    "le": "<=",
    "lt": "<",
}

UNARY_OP: Dict[UnaryOp, str] = {
    "invert": "NOT ",
    "neg": "-",
}


class GenSqlVisitor(Visitor[str]):
    def visit_ArrayNode(self, node: "ArrayNode"):
        elts = ",".join(self(e)[5 if isinstance(e, ArrayNode) else 0 :] for e in node.exprs)
        return f"ARRAY[{elts}]"

    def visit_BinOpNode(self, node: "BinOpNode"):
        left = self(node.left)
        right = self(node.right)
        if node.op != "matmul":
            return f"({left}{BIN_OP[node.op]}{right})"

        if isinstance(node.left, ArrayNode):
            return self(
                ArrayNode([BinOpNode(expr, node.op, node.right) for expr in node.left.exprs])
            )

        if isinstance(node.left, TableAttrNode) and node.left.name == "bbox":
            return f"objectBBox({self(node.left.table.id)}, {right})"

        return f"valueAtTimestamp({left},{right})"

    def visit_BoolOpNode(self, node: "BoolOpNode"):
        op = BOOL_OP[node.op]
        return f"({op.join(self(e) for e in node.exprs)})"

    def visit_CallNode(self, node: "CallNode"):
        fn = node.fn
        return fn(self, node.params)

    def visit_TableAttrNode(self, node: "TableAttrNode"):
        table = node.table
        if isinstance(table, ObjectTableNode):
            return resolve_object_attr(node.name, table.index)
        elif isinstance(table, CameraTableNode):
            return resolve_camera_attr(node.name, table.index)
        else:
            raise Exception("table type not supported")

    def visit_CompOpNode(self, node: "CompOpNode"):
        left = self(node.left)
        right = self(node.right)
        return f"({left}{COMP_OP[node.op]}{right})"

    def visit_LiteralNode(self, node: "LiteralNode"):
        value = node.value
        if isinstance(value, str) and node.python:
            return f"'{value}'"
        else:
            return str(value)

    def visit_UnaryOpNode(self, node: "UnaryOpNode"):
        return f"({UNARY_OP[node.op]}{self(node.expr)})"

    def visit_TableNode(self, node: "TableNode"):
        raise Exception("table type not supported")

    def visit_ObjectTableNode(self, node: "ObjectTableNode"):
        return self(node.traj)

    def visit_CameraTableNode(self, node: "CameraTableNode"):
        return self(node.cam)

    def visit_CastNode(self, node: "CastNode"):
        return f"({self(node.expr)})::{node.to}"


def resolve_object_attr(attr: str, num: Optional[int] = None):
    if num is None:
        return attr
    return f"t{num}.{attr}"


def resolve_camera_attr(attr: str, num: Optional[int] = None):
    if num is None:
        return attr
    return f"c{num}.{attr}"


# TODO: this is duplicate with the one in database.py
TRAJECTORY_COLUMNS: List[Tuple[str, str]] = [
    ("itemId", "TEXT"),
    ("cameraId", "TEXT"),
    ("objectType", "TEXT"),
    ("color", "TEXT"),
    ("trajCentroids", "tgeompoint"),
    ("largestBbox", "stbox"),
    ("itemHeadings", "tfloat"),
]


def map_object(to: int, from_: Optional[int] = None):
    return ",".join(
        f"{resolve_object_attr(attr, from_)} AS {resolve_object_attr(attr, to)}"
        for attr, _ in TRAJECTORY_COLUMNS
    )


CAMERA_COLUMNS: List[Tuple[str, str]] = [
    ("cameraId", "TEXT"),
    ("frameId", "TEXT"),
    ("frameNum", "Int"),
    ("fileName", "TEXT"),
    ("cameraTranslation", "geometry"),
    ("cameraRotation", "real[4]"),
    ("cameraIntrinsic", "real[3][3]"),
    ("egoTranslation", "geometry"),
    ("egoRotation", "real[4]"),
    ("timestamp", "timestamptz"),
    ("cameraHeading", "real"),
    ("egoHeading", "real"),
    ("cameraTranslationAbs", "geometry"),
]


def map_camera(to: int, from_: Optional[int] = None):
    return ",".join(
        f"{resolve_camera_attr(attr, from_)} AS {resolve_camera_attr(attr, to)}"
        for attr, _ in TRAJECTORY_COLUMNS
    )
