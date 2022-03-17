import ast
import datetime

from metadata import metadata_view

common_geo = ["Xmin", "Ymin", "Zmin", "Xmax", "Ymax", "Zmax"]
common_aggregation = ["asMFJSON", common_geo]


# Map to translate ast comparators to SQL comparators
comparator_map = {
    ast.Eq: "==",  # pypika takes in python function, so it should be `==` not `=`
    ast.NotEq: ">=",
    ast.Lt: "<",
    ast.LtE: "<=",
    ast.Gt: ">",
    ast.GtE: ">=",
}

# Map to translate ast propositions to SQL propositions
propositional_map = {ast.And: "AND", ast.Or: "OR"}


def decompile_comparator(comparator, evaluated_var, view):
    # print(evaluated_var)
    # print(ast.dump(comparator))
    result_comparator = ""
    view_context = view
    if isinstance(comparator, ast.Call):
        func_name = comparator.func.id
        result_comparator = func_name + "("
        args = comparator.args
        for arg in args:
            if isinstance(arg, ast.Attribute):
                table_name = arg.value.id
                table_attr = arg.attr
                view_context, table_name, column_name = resolve_default_view(table_attr, view)
                # TODO: else not default
                result_comparator += table_name + "." + column_name
            elif isinstance(arg, ast.Str):
                result_comparator += arg.s
            elif isinstance(arg, ast.Name):
                if arg.id in evaluated_var:
                    result_comparator += evaluated_var[arg.id]
                else:
                    result_comparator += arg.id
            result_comparator += ","
        result_comparator = result_comparator[:-1] + ")"
    elif isinstance(comparator, ast.Attribute):
        table_name = comparator.value.id
        table_attr = comparator.attr
        # TODO: if view == None:
        # TODO: unresolved, dynamically determine the scan views based on both predicates and select
        view_context, table_name, column_name = resolve_default_view(table_attr, view)
        result_comparator = table_name + "." + column_name
    elif isinstance(comparator, ast.Str):
        result_comparator = "'" + comparator.s + "'"
    elif isinstance(comparator, ast.Name):
        if comparator.id in evaluated_var:
            evaluated_variable = evaluated_var[comparator.id]
        else:
            evaluated_variable = comparator.id
        result_comparator = evaluated_variable
    else:
        print(comparator)

    return result_comparator, view_context


def resolve_default_view(attr_name, view):
    view_context = view
    if view is None:
        column_name = metadata_view.trajectory_view.resolve_key(attr_name)
        if column_name:
            view_context = metadata_view.trajectory_view
        else:
            column_name = metadata_view.location_view.resolve_key(attr_name)
            view_context = metadata_view.location_view
        table_name = view_context.view_name
    elif view.default:
        if view.view_name == "metadata_view":
            column_name = view.resolve_key(attr_name)
            table_name = view.map_view(column_name).view_name
        else:
            column_name = view.resolve_key(attr_name)
            if not column_name:
                view_context = metadata_view
                column_name = metadata_view.resolve_key(attr_name)
                table_name = metadata_view.map_view(column_name).view_name
            else:
                table_name = view.view_name

    return view_context, table_name, column_name


def decompile_filter(ast_tree, evaluated_var, view):
    print(ast.dump(ast_tree))
    attributes = []
    operations = []
    comparators = []
    bool_ops = [""]
    cast_types = []
    result_view = view
    for ast_node in ast.walk(ast_tree):
        module_body = ast_node.body[0]
        if isinstance(module_body, ast.Return):
            value = module_body.value
            # if isinstance(value, ast.BoolOp)
            # case where we allow multiple constraints in a single filter, usually for OR
            if isinstance(value, ast.Compare):
                left = value.left
                attribute, left_comebine_view = decompile_comparator(left, evaluated_var, view)
                right = value.comparators[0]
                comparator, right_combine_view = decompile_comparator(right, evaluated_var, view)

                op = value.ops[0]
                if type(op) in comparator_map:
                    operation = comparator_map[type(op)]
                elif isinstance(op, ast.In):
                    if isinstance(comparator, list):
                        operation = " IN "
                    elif isinstance(comparator, str):
                        operation = "overlap"

                if operation == "overlap":
                    attribute = "overlap(%s, %s)" % (attribute, comparator)
                    operation = "="
                    comparator = "true"
                elif operation == " IN ":
                    comparator = list_to_str(comparator)

                attributes.append(attribute)
                operations.append(operation)
                comparators.append(comparator)

        return (
            attributes,
            operations,
            comparators,
            bool_ops,
            cast_types,
            left_comebine_view or right_combine_view,
        )


def list_to_str(lst):
    result = "("
    for s in lst:
        result = result + "'" + s + "'" + ","
    result = result[:-1] + ")"
    return result


def convert_time(start, interval=[]):
    if len(interval) == 0:
        starttime = str(datetime.datetime.min)
        endtime = str(datetime.datetime.max)
    else:
        starttime = str(start + datetime.timedelta(seconds=interval[0]))
        endtime = str(start + datetime.timedelta(seconds=interval[1]))
    return starttime, endtime


def overlap(stbox1, stbox2):
    """Translate the overlap function to psql overlap function"""
    return "Overlap(%s, %s)" % (stbox1, stbox2)


def Tmin(stbox):
    """Translate the Tmin function to psql Tmin function"""
    return "Tmin"


def Tmax(stbox):
    """Translate the Tmax function to psql Tmax function"""
    return "Tmax"


def COUNT(key):
    """SQL Count"""
    return "COUNT(%s)" % key
