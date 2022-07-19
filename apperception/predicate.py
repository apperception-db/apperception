from typing import (Any, Callable, Dict, Generic, Iterable, List, Literal, Optional,
                    Set, Tuple, TypeVar)


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

    def __add__(self, other: "PredicateNode"):
        if not isinstance(other, PredicateNode):
            raise Exception()
        return BinOpNode(self, "add", other)

    def __sub__(self, other: "PredicateNode"):
        if not isinstance(other, PredicateNode):
            raise Exception()
        return BinOpNode(self, "sub", other)

    def __mul__(self, other: "PredicateNode"):
        if not isinstance(other, PredicateNode):
            raise Exception()
        return BinOpNode(self, "mul", other)

    def __truediv__(self, other: "PredicateNode"):
        if not isinstance(other, PredicateNode):
            raise Exception()
        return BinOpNode(self, "div", other)

    def __matmul__(self, other: "PredicateNode"):
        if not isinstance(other, PredicateNode):
            raise Exception()
        return BinOpNode(self, "matmul", other)

    def __and__(self, other: "PredicateNode"):
        if not isinstance(other, PredicateNode):
            raise Exception()
        return BoolOpNode("and", [self, other])

    def __or__(self, other: "PredicateNode"):
        if not isinstance(other, PredicateNode):
            raise Exception()
        return BoolOpNode("or", [self, other])

    def __eq__(self, other: "PredicateNode"):
        if not isinstance(other, PredicateNode):
            raise Exception()
        return CompOpNode(self, "eq", other)

    def __ne__(self, other: "PredicateNode"):
        if not isinstance(other, PredicateNode):
            raise Exception()
        return CompOpNode(self, "ne", other)

    def __ge__(self, other: "PredicateNode"):
        if not isinstance(other, PredicateNode):
            raise Exception()
        return CompOpNode(self, "ge", other)

    def __gt__(self, other: "PredicateNode"):
        if not isinstance(other, PredicateNode):
            raise Exception()
        return CompOpNode(self, "gt", other)

    def __le__(self, other: "PredicateNode"):
        if not isinstance(other, PredicateNode):
            raise Exception()
        return CompOpNode(self, "le", other)

    def __lt__(self, other: "PredicateNode"):
        if not isinstance(other, PredicateNode):
            raise Exception()
        return CompOpNode(self, "lt", other)

    def __invert__(self):
        return UnaryOpNode("invert", self)

    def __neg__(self):
        return UnaryOpNode("neg", self)

    def __repr__(self):
        return f"{self.__class__.__name__}({', '.join(f'{k}={getattr(self, k).__repr__()}' for k in self.__annotations__)})"


class ArrayNode(PredicateNode):
    exprs: List["PredicateNode"]


def arr(*exprs: "PredicateNode"):
    return ArrayNode(exprs)


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
    python: bool = True


class TableNode(PredicateNode):
    index: Optional[int]
    _short = {}

    def __getattr__(self, name: str) -> "TableAttrNode":
        return TableAttrNode(self._short.get(name, name), self, False)

    def __repr__(self):
        if self.index is None:
            return self.__class__.__name__
        return f"{self.__class__.__name__}[{self.index}]"


class ObjectTableNode(TableNode):
    def __init__(self, index: int):
        self.index = index
        self.traj = TableAttrNode("trajCentroids", self, True)
        self.trans = TableAttrNode("translations", self, True)
        self.id = TableAttrNode("itemId", self, True)


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


Fn = Callable[["Visitor", Iterable["PredicateNode"]], str]


class CallNode(PredicateNode):
    fn: "Fn"
    params: List["PredicateNode"]


def call_node(fn: "Fn"):
    def call_node_factory(*args: "PredicateNode") -> "CallNode":
        return CallNode(fn, list(args))
    return call_node_factory


T = TypeVar("T")


class Visitor(Generic[T]):
    def __call__(self, node: "PredicateNode") -> T:
        attr = f"visit_{node.__class__.__name__}"
        if not hasattr(self, attr):
            raise Exception("Unknown node type:", node.__class__.__name__)
        return getattr(self, attr)(node)

    def visit_ArrayNode(self, node: "ArrayNode") -> T:
        for e in node.exprs:
            self(e)

    def visit_CompOpNode(self, node: "CompOpNode") -> T:
        self(node.left)
        self(node.right)

    def visit_BinOpNode(self, node: "BinOpNode") -> T:
        self(node.left)
        self(node.right)

    def visit_BoolOpNode(self, node: "BoolOpNode") -> T:
        for e in node.exprs:
            self(e)

    def visit_UnaryOpNode(self, node: "UnaryOpNode") -> T:
        self(node.expr)

    def visit_LiteralNode(self, node: "LiteralNode") -> T:
        ...

    def visit_TableAttrNode(self, node: "TableAttrNode") -> T:
        self(node.table)

    def visit_CallNode(self, node: "CallNode") -> T:
        for p in node.params:
            self(p)

    def visit_TableNode(self, node: "TableNode") -> T:
        ...

    def visit_ObjectTableNode(self, node: "ObjectTableNode") -> T:
        ...

    def visit_CameraTableNode(self, node: "CameraTableNode") -> T:
        ...


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
        return TableAttrNode(node.name, self(node.table))

    def visit_CallNode(self, node: "CallNode"):
        return CallNode(node.fn, [self(p) for p in node.params])

    def visit_TableNode(self, node: "TableNode"):
        return node

    def visit_ObjectTableNode(self, node: "ObjectTableNode"):
        return node

    def visit_CameraTableNode(self, node: "CameraTableNode"):
        return node


class ExpandBoolOpTransformer(BaseTransformer):
    def visit(self, node: "PredicateNode"):
        if isinstance(node, BoolOpNode):
            exprs: List["PredicateNode"] = []
            for expr in node.exprs:
                e = self(expr)
                if isinstance(e, BoolOpNode) and e.op == node.op:
                    exprs.extend(e.exprs)
                else:
                    exprs.append(e)
            return BoolOpNode(node.op, exprs)
        return super()(node)


class FindAllTablesVisitor(Visitor[None]):
    tables: Set[int]
    camera: bool

    def __init__(self):
        self.tables = set()
        self.camera = False

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
            return ObjectTableNode(objects[self.mapping[node.index]])
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
                ArrayNode([BinOpNode(l, node.op, node.right) for l in node.left.exprs])
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

    def visit_ObjectTableNode(self, node: "ObjectTableNode") -> T:
        return self(node.traj)

    def visit_CameraTableNode(self, node: "CameraTableNode") -> T:
        return self(node.cam)


def resolve_object_attr(attr: str, num: Optional[int] = None):
    if num is None:
        return attr
    return f"t{num}_{attr}"


def resolve_camera_attr(attr: str, num: Optional[int] = None):
    if num is None:
        return attr
    return f"c{num}_{attr}"


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
