from __future__ import annotations

import ast
import copy
import os
from typing import Callable, List, Optional

import uncompyle6
from metadata import MetadataView, View, metadata_view
from metadata_util import (COUNT, Tmax, Tmin, common_aggregation, common_geo,
                           convert_time, decompile_filter)


class Project:
    # TODO: Add checks for names
    # Select Node (contains Column Nodes and Aggregate Nodes
    # within Column Nodes)

    def __init__(self, root):
        self.root = root
        self.distinct = False
        self.column_nodes = []

    def append(self, column_node):
        self.column_nodes.append(column_node)

    def find(self, column_name):
        for column_node in self.column_nodes:
            if column_node.column_name == column_name:
                return column_node
        return None

    def remove(self, column_name):
        column_node = self.find(column_name)
        self.column_nodes.remove(column_node)

    def is_empty(self):
        return len(self.column_nodes) == 0


class Column:
    def __init__(self, column_name: str):
        self.column_name: str = column_name
        self.aggr_nodes: List[Aggregate] = []

    def aggregate(self, func_name: str, parameters: List[str] = [], special_args: List[str] = []):
        if func_name in common_aggregation:
            if len(special_args) > 0:
                agg_node = eval(func_name)(func_name, parameters, special_args)
            else:
                agg_node = eval(func_name)(func_name, parameters)
        else:
            agg_node = Aggregate(func_name, parameters)
        self.aggr_nodes.append(agg_node)
        return self

    def get_coordinates(self):
        # self.aggregate("asMFJSON", special_args=["coordinates"])
        self.aggregate("asMFJSON")

    def interval(self, starttime, endtime):
        self.aggregate("atPeriodSet", parameters=["'{[%s, %s)}'" % (starttime, endtime)])


class Aggregate:
    def __init__(self, func_name: str, parameters: list = []):
        self.func_name = func_name
        self.parameters = parameters


class asMFJSON(Aggregate):
    def __init__(self, func_name="asMFJSON", parameters: list = [], interesting_fields=[]):
        super().__init__(func_name, parameters)
        self.interesting_fields = interesting_fields

    # def function_map(self):


class Scan:
    def __init__(self, root):
        self.view: Optional[View] = None
        self.root = root

    def add_view(self, view: View):
        self.view = view


class Filter:
    def __init__(self, root):
        self.predicates = []
        self.root = root

    def append(self, predicate):
        self.predicates.append(predicate)
        predicate.root = self
        predicate.decompile()
        return self.root.view(use_view=predicate.view_context)

    def is_empty(self):
        return len(self.predicates) == 0

    def get_view(self):
        return self.root.scan.view


class Predicate:
    def __init__(self, predicate: Callable[[int], bool], evaluated_var={}):
        self.predicate = predicate
        s = uncompyle6.deparse_code2str(self.predicate.__code__, out=open(os.devnull, "w"))
        self.t = ast.parse(s)
        self.evaluated_var = evaluated_var
        self.root = None

    def decompile(self):
        # assert self.root
        (
            self.attribute,
            self.operation,
            self.comparator,
            self.bool_ops,
            self.cast_types,
            self.view_context,
        ) = decompile_filter(self.t, self.evaluated_var, self.root.get_view())

    def new_decompile(self):
        (
            self.attribute,
            self.operation,
            self.comparator,
            self.bool_ops,
            self.cast_types,
            self.view_context,
        ) = decompile_filter(self.t, self.evaluated_var, None)

    def get_compile(self):
        return self.attribute, self.operation, self.comparator, self.bool_ops, self.cast_types


class Group:
    def __init__(self, root):
        self.group = None


class MetadataContext:
    """Context Root Node"""

    def __init__(self, single_mode=True):
        # Initialize the root, which is itself
        self.root = self
        self.start_time = None
        self.project = Project(self.root)
        self.scan = Scan(self.root)
        self.filter = Filter(self.root)
        self.groupby = None
        self.single_mode = single_mode
        # self.orderby_nodes = [orderby_node1, orderby_node2...] # we dont need these for now

    def select_column(self, column_key):
        """Select a specific column"""
        mapped_view = metadata_view.map_view(column_key)
        if self.scan.view is None:
            self.scan.view = mapped_view
        elif (
            self.scan.view.default
            and mapped_view.default
            and self.scan.view.view_name != mapped_view.view_name
        ):
            self.scan.view = metadata_view

        view_name = mapped_view.view_name
        column_node = Column(view_name + "." + column_key)
        self.project.append(column_node)
        return column_node

    def delete_column(self, column_name):
        """Remove column in column nodes in question"""
        self.project.remove(column_name)

    def clear(self):
        """Restart a context from scratch"""
        self.project = Project(self.root)
        self.scan = Scan(self.root)
        self.filter = Filter(self.root)

    def get_columns(self, *argv, distinct=False):
        if not self.single_mode:
            self.project.distinct = distinct
            for arg in argv:
                arg(self)
            return self
        else:
            new_context = copy.deepcopy(self)
            new_context.project.distinct = distinct
            for arg in argv:
                new_context = arg(new_context)
            return new_context

    # The following functions would be Apperception commands
    def predicate(self, p, evaluated_var={}):
        if not self.single_mode:
            new_predicate = Predicate(p, evaluated_var)
            self.filter.append(new_predicate)
            return self
        else:
            # make a copy of self first
            new_context = copy.deepcopy(self)

            new_predicate = Predicate(p, evaluated_var)
            new_context = new_context.filter.append(new_predicate)
            return new_context

    def selectkey(self, distinct=False):
        if not self.single_mode:
            self.project.distinct = distinct
            # self.select_column(MetadataView.camera_id)
            self.select_column(MetadataView.object_id)
            return self
        else:
            # make a copy of self first
            new_context = copy.deepcopy(self)
            new_context.project.distinct = distinct

            # new_context.select_column(MetadataView.camera_id)
            new_context.select_column(MetadataView.object_id)
            return new_context

    def get_object_type(self, distinct=False):
        if not self.single_mode:
            self.project.distinct = distinct
            # self.select_column(MetadataView.camera_id)
            self.select_column(MetadataView.object_type)
            return self
        else:
            # make a copy of self first
            new_context = copy.deepcopy(self)
            new_context.project.distinct = distinct

            # new_context.select_column(MetadataView.camera_id)
            new_context.select_column(MetadataView.object_type)
            return new_context

    def get_trajectory(self, time_interval=[], distinct=False):
        # TODO: return a proxy type
        if not self.single_mode:
            self.project.distinct = distinct
            traj_column = self.select_column(MetadataView.trajectory)
            starttime, endtime = convert_time(self.start_time, time_interval)
            traj_column.interval(starttime, endtime)
            traj_column.get_coordinates()
            return self
        else:
            # make a copy of self first
            new_context = copy.deepcopy(self)
            new_context.project.distinct = distinct
            traj_column = new_context.select_column(MetadataView.trajectory)
            starttime, endtime = convert_time(self.start_time, time_interval)
            traj_column.interval(starttime, endtime)
            traj_column.get_coordinates()
            return new_context

    def get_geo(self, time_interval=[], distinct=False):
        # TODO: return a proxy type
        if not self.single_mode:
            self.project.distinct = distinct
            for geo_func in common_geo:
                new_trajColumn = self.select_column(MetadataView.location)
                new_trajColumn.aggregate(geo_func)

            self.interval(time_interval)
            return self
        else:
            # make a copy of self first
            new_context = copy.deepcopy(self)
            new_context.project.distinct = distinct
            for geo_func in common_geo:
                new_trajColumn = new_context.select_column(MetadataView.location)
                new_trajColumn.aggregate(geo_func)

            new_context.interval(time_interval)
            return new_context

    def interval(self, time_interval):
        # TODO: return a proxy type
        start, end = convert_time(self.start_time, time_interval)
        if not self.single_mode:
            self.predicate(lambda obj: Tmin(obj.location) >= start, {"start": "'" + start + "'"})
            self.predicate(lambda obj: Tmax(obj.location) < end, {"end": "'" + end + "'"})
            return self
        else:
            new_context = self.predicate(
                lambda obj: Tmin(obj.location) >= start, {"start": "'" + start + "'"}
            ).predicate(lambda obj: Tmax(obj.location) < end, {"end": "'" + end + "'"})
            return new_context

    def get_time(self, distinct=False):
        # TODO: return a proxy type
        if not self.single_mode:
            self.project.distinct = distinct
            new_trajColumn = self.select_column(MetadataView.location)
            new_trajColumn.aggregate("Tmin")
            return self
        else:
            # make a copy of self first
            new_context = copy.deepcopy(self)
            new_context.project.distinct = distinct
            new_trajColumn = new_context.select_column(MetadataView.location)
            new_trajColumn.aggregate("Tmin")
            return new_context

    def get_distance(self, time_interval=[], distinct=False):
        # TODO: return a proxy type
        if not self.single_mode:
            self.project.distinct = distinct
            traj_column = self.select_column(MetadataView.trajectory)
            starttime, endtime = convert_time(self.start_time, time_interval)
            traj_column.interval(starttime, endtime)
            traj_column.aggregate("cumulativeLength")
            return self
        else:
            # make a copy of self first
            new_context = copy.deepcopy(self)
            new_context.project.distinct = distinct
            starttime, endtime = convert_time(self.start_time, time_interval)
            traj_column.interval(starttime, endtime)
            traj_column.aggregate("cumulativeLength")
            return new_context

    def get_speed(self, time_interval=[], distinct=False):
        # TODO: return a proxy type
        if not self.single_mode:
            self.project.distinct = distinct
            traj_column = self.select_column(MetadataView.trajectory)
            starttime, endtime = convert_time(self.start_time, time_interval)
            traj_column.interval(starttime, endtime)
            traj_column.aggregate("speed")
            return self
        else:
            # make a copy of self first
            new_context = copy.deepcopy(self)
            new_context.project.distinct = distinct
            traj_column = new_context.select_column(MetadataView.trajectory)
            starttime, endtime = convert_time(self.start_time, time_interval)
            traj_column.interval(starttime, endtime)
            traj_column.aggregate("speed")
            return new_context

    def count(self, key):
        # make a copy of self first
        new_context = copy.deepcopy(self)

        count_map = {
            MetadataContext.get_trajectory: "trajCentroids",
            MetadataContext.get_time: "Tmin(trajBbox)",
            MetadataContext.selectkey: "distinct(cameraId, itemId)",
        }
        traj_column = new_context.select_column(count_map[key])
        traj_column.aggregate(COUNT)
        return new_context

    def group(self, key):
        # make a copy of self first
        new_context = copy.deepcopy(self)
        new_context.groupby = Group(key)

    def view(self, view_name="", use_view=None):
        # TODO:Not fully functioned yet
        if not self.single_mode:
            if use_view:
                self.scan.add_view(use_view)
            else:
                temp_view = View(view_name)
                temp_view.context = self
                self.scan.add_view(temp_view)
            return self
        else:
            # make a copy of self first
            new_context = copy.deepcopy(self)
            if use_view:
                new_context.scan.add_view(use_view)
            else:
                temp_view = View(view_name)
                temp_view.context = self
                new_context.scan.add_view(temp_view)
                # need to figure out the return value of the view command;
            return new_context

    def join(self, join_view, join_type="", join_condition=""):
        # make a copy of self first
        new_context = copy.deepcopy(self)

        if join_view.view_name == metadata_view.view_name:
            new_context.scan.join(metadata_view.trajectory_view)
            new_context.scan.join(metadata_view.location_view)
        else:
            new_context.scan.join(join_view)

        return new_context


primarykey = MetadataContext.selectkey
trajectory = MetadataContext.get_trajectory
distance = MetadataContext.get_distance
speed = MetadataContext.get_speed
geometry = MetadataContext.get_geo
object_type = MetadataContext.get_object_type
time = MetadataContext.get_time
