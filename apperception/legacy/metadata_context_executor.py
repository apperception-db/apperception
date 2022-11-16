import numpy as np
import psycopg2

from apperception.data_types.views import View, metadata_view
from apperception.legacy.metadata_context import (
    Aggregate,
    Column,
    Filter,
    MetadataContext,
    Predicate,
    Project,
    Scan,
    asMFJSON,
)
from apperception.legacy.metadata_util import common_aggregation
from apperception.utils import join


class MetadataContextExecutor:
    """Executor class to execute the context input
    Essentially translates the context to a SQL query that
    the backend and interpret
    """

    def __init__(self, conn, new_context: MetadataContext = None):
        if new_context:
            self.context(new_context)
        self.conn = conn

    def connect_db(
        self, host="localhost", user=None, password=None, port=25432, database_name=None
    ):
        """Connect to the database"""
        self.conn = psycopg2.connect(
            database=database_name, user=user, password=password, host=host, port=port
        )

    def context(self, new_context: MetadataContext):
        self.current_context = new_context
        return self

    def visit(self, create_view: bool, view_name: str):
        select_query = self.visit_project(self.current_context.project)
        from_query = self.visit_scan(self.current_context.scan)
        where_query = self.visit_filter(self.current_context.filter)
        if create_view:
            db_query = (
                "CREATE VIEW " + view_name + " AS " + select_query + from_query + where_query + ";"
            )
            print(db_query + "\n")
            return "SELECT * FROM " + view_name + ";"
        else:
            db_query = select_query + from_query + where_query + ";"
            print(db_query + "\n")
            return db_query

    def visit_project(self, project_node: Project):
        select_query: str = "SELECT "
        if project_node.distinct:
            select_query += "distinct on(itemId) "
        if project_node.is_empty():
            return select_query + "* "
        for column_node in project_node.column_nodes:
            select_query += self.visit_column(column_node)
            select_query += ", "
        select_query = select_query[:-2]
        return select_query

    def visit_scan(self, scan_node: Scan):
        from_query: str = " From "
        if scan_node.view:
            if scan_node.view.default:
                if scan_node.view == metadata_view:
                    from_query += (
                        metadata_view.trajectory_view.view_name
                        + " INNER JOIN "
                        + metadata_view.location_view.view_name
                        + " USING(itemId) "
                    )
                else:
                    from_query = from_query + scan_node.view.view_name + " "
        # for view_node in scan_node.views:
        #     from_query += self.visit_table(view_node)
        #     from_query += ", "
        # from_query = from_query[:-2]
        return from_query

    def visit_filter(self, filter_node: Filter):
        where_query = " Where "
        if filter_node.is_empty():
            return ""
        for predicate_node in filter_node.predicates:
            where_query += self.visit_predicate(predicate_node)
            where_query += " AND "
        where_query = where_query[:-5]
        return where_query

    def visit_column(self, column_node: Column):
        aggregated = column_node.column_name
        for aggr_node in column_node.aggr_nodes:
            aggregated = translate_aggregation(aggr_node, aggregated)
            print(aggregated)
        return aggregated

    def visit_table(self, view_node: View):
        return view_node.view_name

    def visit_predicate(self, predicate_node: Predicate):
        attribute, operation, comparator, bool_ops, cast_types = predicate_node.get_compile()
        # assert(len(attribute) == len(operation) == len(comparator) == len(bool_ops) == len(cast_types))
        predicate_query = ""
        for i in range(len(attribute)):
            attr = attribute[i]
            op = operation[i]
            comp = comparator[i]
            bool_op = bool_ops[i]
            # cast_type = cast_types[i]
            # cast_str = "::" + cast_type if cast_type != "" else ""
            # predicate_query += bool_op + attr + cast_str + op + comp + cast_str
            predicate_query += bool_op + attr + op + comp
        return predicate_query

    def execute(self, create_view: bool = False, view_name: str = ""):
        self.cursor = self.conn.cursor()
        self.cursor.execute(self.visit(create_view=create_view, view_name=view_name))
        return np.asarray(self.cursor.fetchall())


def translate_aggregation(aggr_node: Aggregate, aggregated: str):
    aggregated = f"{aggr_node.func_name}({join([aggregated, *aggr_node.parameters])})"

    if isinstance(aggr_node, asMFJSON) and aggr_node.func_name in common_aggregation:
        if len(aggr_node.interesting_fields) > 0:
            interesting_field = aggr_node.interesting_fields[0]
            aggregated += f"::json->'{interesting_field}'"
        else:
            aggregated += "::json"
    return aggregated
