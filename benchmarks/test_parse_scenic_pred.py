import ast
import os

from decompyle3 import deparse_code2str
from astpretty import pprint as apprint


def main():
    def pred(obj, cam):
        return (cam.x - 10) <= obj.x <= (cam.x + 10) and (cam.y - 15) <= obj.y <= (cam.y + 70)

    s = deparse_code2str(pred.__code__, out=open(os.devnull, "w"))
    tree = ast.parse(s)
    # print(pred.__code__)
    # print(s)
    # apprint(tree)
    subtree = tree.body[0]
    assert isinstance(subtree, ast.Return)

    apprint(subtree)

    x_range = []
    y_range = []

    if isinstance(subtree.value, ast.BoolOp):
        left_node = subtree.value.values[0]
        right_node = subtree.value.values[1]

        # parse left
        if isinstance(left_node, ast.Compare):
            cmp_node = left_node
            left = cmp_node.left
            # ops = cmp_node.ops
            comparators = cmp_node.comparators

            if (
                len(comparators) == 2
                and isinstance(comparators[0], ast.Attribute)
                and comparators[0].attr == "x"
            ):
                if isinstance(left, ast.BinOp):
                    if isinstance(left.left, ast.Attribute) and left.left.attr == "x":
                        if isinstance(left.op, ast.Sub):
                            assert isinstance(left.right, ast.Num)
                            x_range.append(-left.right.n)
                        elif isinstance(left.op, ast.Add):
                            assert isinstance(left.right, ast.Num)
                            x_range.append(left.right.n)

                if isinstance(comparators[-1], ast.BinOp):
                    right = comparators[-1]
                    if isinstance(right.left, ast.Attribute) and right.left.attr == "x":
                        if isinstance(right.op, ast.Sub):
                            assert isinstance(right.right, ast.Num)
                            x_range.append(-right.right.n)
                        elif isinstance(right.op, ast.Add):
                            assert isinstance(right.right, ast.Num)
                            x_range.append(right.right.n)

        if isinstance(right_node, ast.Compare):
            cmp_node = right_node
            left = cmp_node.left
            # ops = cmp_node.ops
            comparators = cmp_node.comparators

            if (
                len(comparators) == 2
                and isinstance(comparators[0], ast.Attribute)
                and comparators[0].attr == "y"
            ):
                if isinstance(left, ast.BinOp):
                    if isinstance(left.left, ast.Attribute) and left.left.attr == "y":
                        if isinstance(left.op, ast.Sub):
                            assert isinstance(left.right, ast.Num)
                            y_range.append(-left.right.n)
                        elif isinstance(left.op, ast.Add):
                            assert isinstance(left.right, ast.Num)
                            y_range.append(left.right.n)

                if isinstance(comparators[-1], ast.BinOp):
                    right = comparators[-1]
                    if isinstance(right.left, ast.Attribute) and right.left.attr == "y":
                        if isinstance(right.op, ast.Sub):
                            assert isinstance(right.right, ast.Num)
                            y_range.append(-right.right.n)
                        elif isinstance(right.op, ast.Add):
                            assert isinstance(right.right, ast.Num)
                            y_range.append(right.right.n)

    print(x_range)
    print(y_range)


if __name__ == "__main__":
    main()
