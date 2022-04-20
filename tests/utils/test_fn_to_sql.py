import pytest
from apperception.utils.fn_to_sql import fn_to_sql


@pytest.mark.parametrize("fn, sql", [
    (lambda o, _ : o.c1 + 1, "(T.c1+1)"),
    (lambda o, c : o.c1 + c.c1, "(T.c1+C.c1)"),
    (lambda o, c : o.c1 - c.c1, "(T.c1-C.c1)"),
    (lambda o, c : o.c1 * c.c1, "(T.c1*C.c1)"),
    (lambda o, c : o.c1 / c.c1, "(T.c1/C.c1)"),
    (lambda o, c : o.c1 % c.c1, "(T.c1%C.c1)"),
    (lambda o, c : o.c1 == c.c1, "(T.c1=C.c1)"),
    (lambda o, c : o.c1 < c.c1, "(T.c1<C.c1)"),
    (lambda o, c : o.c1 > c.c1, "(T.c1>C.c1)"),
    (lambda o, c : o.c1 <= c.c1, "(T.c1<=C.c1)"),
    (lambda o, c : o.c1 >= c.c1, "(T.c1>=C.c1)"),
    (lambda o, c : o.c1 != c.c1, "(T.c1<>C.c1)"),
    (lambda o, c : o.c1 in c.c1, "(T.c1 IN C.c1)"),
    (lambda o, c : +o.c1, "(+T.c1)"),
    (lambda o, c : -o.c1, "(-T.c1)"),
    (lambda o, c : not o.c1, "(NOT T.c1)"),
    (lambda o, c : o.c1 and o.c2 and o.c3, "(T.c1 AND T.c2 AND T.c3)"),
    (lambda o, c : o.c1 or o.c2 or o.c3, "(T.c1 OR T.c2 OR T.c3)"),
])
def test_simple_ops(fn, sql):
    assert fn_to_sql(fn, ["T", "C"]) == sql


@pytest.mark.parametrize("fn, sql", [
    (lambda o, c : (o.c1 + c.c1) - c.c2 + o.c2 * c.c3 / o.c3, "(((T.c1+C.c1)-C.c2)+((T.c2*C.c3)/T.c3))"),
    (lambda o, c : o.c1 == c.c1 and (o.c2 < c.c2 or o.c3 == c.c3), "((T.c1=C.c1) AND ((T.c2<C.c2) OR (T.c3=C.c3)))"),
])
def test_nested(fn, sql):
    assert fn_to_sql(fn, ["T", "C"]) == sql


@pytest.mark.parametrize("fn, sql", [
    (lambda o, c : [o.c1, c.c1, 3], "ARRAY[T.c1,C.c1,3]"),
    (lambda o, c : c.c1 in o.c1[1:2], "(C.c1 IN T.c1[1:2])"),
    (lambda o, c : c.c1[o.c1[3]] in o.c1[1:2], "(C.c1[T.c1[3]] IN T.c1[1:2])"),
])
def test_array(fn, sql):
    assert fn_to_sql(fn, ["T", "C"]) == sql


@pytest.mark.parametrize("fn, sql", [
    ("lambda o, c : test(o.c1, o.c2, o.c3)", "test(T.c1, T.c2, T.c3)"),
    ("lambda o, c : test(o.c1) + 3", "(test(T.c1)+3)"),
])
def test_fn(fn, sql):
    assert fn_to_sql(fn, ["T", "C"]) == sql


def test_def():
    def test1(o, c):
        return o.c1 + c.c2
    assert fn_to_sql(test1, ["T", "C"]) == "(T.c1+C.c2)"
