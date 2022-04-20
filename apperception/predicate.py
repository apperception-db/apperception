from __future__ import annotations
import ast
import os
from typing import Any, Callable, Dict, List, Union
from sys import version_info
from inspect import getfullargspec, FullArgSpec

if version_info.major != 3:
    raise Exception("Only support python3")
if version_info.minor < 7:
    from uncompyle6 import deparse_code2str
else:
    from decompyle3 import deparse_code2str

from . import F


POSTGRES_FUNC: Dict[str, Callable[[PredicateVisitor, List[str]], str]] = {
    "convert_camera": F.convert_camera.fn
}


class Predicate:
    _ast: ast.Module
    _vars: Dict[str, Any]

    def __init__(self, predicate: Union[str, Callable], eval_vars: Dict[str, Any] = {}):
        if not isinstance(predicate, str):
            predicate = to_lambda_str(predicate)

        self._ast = ast.parse(predicate)
        self._vars = eval_vars

    def to_sql(self, tables: List[str], eval_vars: Dict[str, Any] = {}):
        expr = self._ast.body[0]

        if not isinstance(expr, ast.Expr):
            raise Exception("Predicate should produce an ast.Expr: ", expr)
        return PredicateVisitor(tables, {**self._vars, **eval_vars}).visit(expr.value)


def to_lambda_str(predicate: Callable) -> str:
    """
    Parse a lambda or a one-line def-function into a string of lambda
    The output from deparse_code2str does not include function signture.
    For example, lambda x : x + 1 will be deparsed into 'return x + 1'
    """
    argspec = getfullargspec(predicate)
    predicate_str = deparse_code2str(predicate.__code__, out=open(os.devnull, "w"))
    if not validate(predicate_str, argspec):
        raise Exception()
    return f"lambda {', '.join(argspec.args)} :{predicate_str[len('return '):]}"


def validate(predicate: str, argspec: FullArgSpec):
    return (
        isinstance(predicate, str)
        and predicate.startswith("return ")
        and len([*filter(lambda line: line != "", predicate.splitlines())]) == 1
        and argspec.varargs is None
        and argspec.varkw is None
        and argspec.defaults is None
        and len(argspec.kwonlyargs) == 0
    )


class PredicateVisitor(ast.NodeVisitor):
    tables: List[str]
    eval_vars: Dict[str, Any]

    def __init__(self, tables: List[str], eval_vars: Dict[str, Any]) -> None:
        super().__init__()
        self.tables = tables
        self.eval_vars = eval_vars

    def visit(self, node: ast.AST) -> str:
        return super().visit(node)

    def generic_visit(self, node: ast.AST) -> None:
        raise Exception("Unsupported node type: " + type(node).__name__)

    def visit_Lambda(self, node: ast.Lambda) -> str:
        args = [arg.arg for arg in node.args.args]
        if len(args) != len(self.tables):
            raise Exception("The number of tables should match the number of predicate arguments")

        for arg, table in zip(args, self.tables):
            self.eval_vars[arg] = table
        return self.visit(node.body)

    def visit_BoolOp(self, node: ast.BoolOp) -> str:
        op = bool_op(node.op)
        return f"({op.join([*map(self.visit, node.values)])})"

    def visit_BinOp(self, node: ast.BinOp) -> str:
        left = self.visit(node.left)
        op = bin_op(node.op)
        right = self.visit(node.right)
        return f"({left}{op}{right})"

    def visit_UnaryOp(self, node: ast.UnaryOp) -> str:
        op = unary_op(node.op)
        operand = self.visit(node.operand)
        return f"({op}{operand})"

    def visit_Compare(self, node: ast.Compare) -> str:
        ops = node.ops
        if len(ops) > 1 or len(node.comparators) > 1:
            raise Exception("Does not support multiple ops")
        left = self.visit(node.left)
        op = cmp_op(ops[0])
        right = self.visit(node.comparators[0])
        return f"({left}{op}{right})"

    def visit_Call(self, node: ast.Call) -> str:
        if len(node.keywords) > 0:
            raise Exception("Keyworded argument is not supported")
        if any(isinstance(arg, ast.Starred) for arg in node.args):
            raise Exception("Starred argument is not supported")

        if isinstance(node.func, ast.Attribute):
            value = node.func.value
            if not isinstance(value, ast.Name) or value.id != "F":
                raise Exception("Only allow custom functions from fn module")
            func = node.func.attr
        elif isinstance(node.func, ast.Name):
            func = node.func.id
        else:
            raise Exception("Unsupported function")

        if func not in POSTGRES_FUNC:
            raise Exception("Unsupported function: ", func)

        # args = [self.visit(arg) for arg in node.args]
        return POSTGRES_FUNC[func](self, node.args)

    def visit_Constant(self, node: ast.Constant) -> str:
        value = node.value
        if isinstance(value, str):
            return f"'{value}'"
        return str(value)

    def visit_Attribute(self, node: ast.Attribute) -> str:
        # Should we not allow users to directly acces the database's field?
        value = self.visit(node.value)
        attr = node.attr
        return f"{value}.{attr}"

    def visit_Subscript(self, node: ast.Subscript) -> str:
        if not isinstance(node.slice, ast.Slice) and not isinstance(node.slice, ast.Index):
            raise Exception("Slice must be a slice or an index")

        value = self.visit(node.value)
        slice = self.visit(node.slice)
        return f"{value}[{slice}]"

    def visit_Name(self, node: ast.Name) -> str:
        return self.eval_vars[node.id]

    def visit_List(self, node: ast.List) -> str:
        elts = ','.join(
            self.visit(elt)[5 if isinstance(elt, ast.List) else 0:]
            for elt
            in node.elts
        )
        return f"ARRAY[{elts}]"

    def visit_Slice(self, node: ast.Slice) -> str:
        if node.lower is None or node.upper is None:
            raise Exception("Slice must have lower and upper")
        if node.step is not None:
            raise Exception("Slice step is not supported")

        lower = self.visit(node.lower)
        upper = self.visit(node.upper)
        return f"{lower}:{upper}"

    def visit_Index(self, node: ast.Index) -> str:
        return self.visit(node.value)


def cmp_op(op: ast.cmpop) -> str:
    if isinstance(op, ast.Eq):
        return "="
    if isinstance(op, ast.NotEq):
        return "<>"
    if isinstance(op, ast.Lt):
        return "<"
    if isinstance(op, ast.LtE):
        return "<="
    if isinstance(op, ast.Gt):
        return ">"
    if isinstance(op, ast.GtE):
        return ">="
    if isinstance(op, ast.In):
        return " IN "
    if isinstance(op, ast.NotIn):
        raise Exception("'x not in y' is not supported, use 'not (x in y)' instead")
    raise Exception("Operation not supported: ", op)


def unary_op(op: ast.unaryop) -> str:
    if isinstance(op, ast.UAdd):
        return '+'
    if isinstance(op, ast.USub):
        return '-'
    if isinstance(op, ast.Not):
        return 'NOT '
    raise Exception("Operation not supported: ", op)


def bool_op(op: ast.boolop) -> str:
    if isinstance(op, ast.And):
        return ' AND '
    if isinstance(op, ast.Or):
        return ' OR '
    raise Exception("Operation not supported: ", op)


def bin_op(op: ast.operator) -> str:
    if isinstance(op, ast.Add):
        return '+'
    if isinstance(op, ast.Div):
        return '/'
    if isinstance(op, ast.Mod):
        return '%'
    if isinstance(op, ast.Mult):
        return '*'
    if isinstance(op, ast.Sub):
        return '-'
    raise Exception("Operation not supported: ", op)
