from typing import Callable

import numpy as np
import numpy.typing as npt
from bitarray import bitarray
from postgis import MultiPoint
from psycopg2 import sql
from pyquaternion import Quaternion

from apperception.database import database
from apperception.predicate import (
    ArrayNode,
    BaseTransformer,
    BinOpNode,
    BoolOpNode,
    CallNode,
    CameraTableNode,
    CastNode,
    CompOpNode,
    LiteralNode,
    ObjectTableNode,
    PredicateNode,
    TableAttrNode,
    TableNode,
    UnaryOpNode,
    Visitor,
)
from apperception.utils import F

from ...payload import Payload
from ..stage import Stage

OTHER_ROAD_TYPES = {
    'roadsection': ['road', 'intersection'],
    "lane": ['lanegroup', 'road', 'roadsection', 'intersection'],
    "lanesection": ['lanegroup', 'road', 'roadsection', 'intersection'],
    "lanegroup": ['road', 'roadsection', 'intersection'],
}


class InView(Stage):
    def __init__(self, distance: float, segment_type: "str | list[str]"):
        super().__init__()
        self.distance = distance
        self.segment_types = segment_type if isinstance(segment_type, list) else [segment_type]

    def _run(self, payload: "Payload") -> "tuple[bitarray, None]":
        w, h = payload.video.dimension
        Z = self.distance
        view_vertices_2d = np.array([
            # 4 corners of the image frame
            (w, h, 1),
            (w, 0, 1),
            (0, h, 1),
            (0, 0, 1),
            # camera position
            (0, 0, 0),
        ]).T
        assert view_vertices_2d.shape == (3, 5), view_vertices_2d.shape

        [[fx, s, x0], [_, fy, y0], [_, _, _]] = payload.video.interpolated_frames[0].camera_intrinsic

        # 3x3 matrix to convert points from pixel-coordinate to camera-coordinate
        pixel2camera = Z * np.array([
            [1 / fx, -s / (fx * fy), (s * y0 / (fx * fy)) - (x0 / fx)],
            [0, 1 / fy, -y0 / fy],
            [0, 0, 1]
        ])
        assert pixel2camera.shape == (3, 3), pixel2camera.shape

        view_vertices_from_camera = pixel2camera @ view_vertices_2d
        assert view_vertices_from_camera.shape == (3, 5), view_vertices_from_camera.shape

        extrinsics: "list[npt.NDArray]" = []
        indices: "list[int]" = []
        for i, (k, f) in enumerate(zip(payload.keep, payload.video.interpolated_frames)):
            if not k:
                continue

            rotation = Quaternion(f.camera_rotation)
            rotation_matrix = rotation.unit.rotation_matrix
            assert rotation_matrix.shape == (3, 3), rotation_matrix.shape

            # 3x4 matrix to convert points from camera-coordinate to world-coordinate
            translation = np.array(f.camera_translation)[np.newaxis].T
            extrinsic = np.hstack((rotation_matrix, translation))
            assert extrinsic.shape == (3, 4), extrinsic.shape

            extrinsics.append(extrinsic)
            indices.append(i)

        N = len(extrinsics)

        # add 1 to the last row
        view_vertices_from_camera = np.concatenate((
            view_vertices_from_camera,
            np.ones_like(view_vertices_from_camera[:1]),
        ))

        _extrinsics = np.stack(extrinsics)
        assert _extrinsics.shape == (N, 3, 4), _extrinsics.shape

        # convert 4 corner points from pixel-coordinate to world-coordinate
        view_area_3ds = _extrinsics @ view_vertices_from_camera
        assert view_area_3ds.shape == (N, 3, 5), view_area_3ds.shape

        # project view_area to 2D from top-down view
        view_area_2ds = view_area_3ds[:, :2].swapaxes(1, 2)
        assert view_area_2ds.shape == (N, 5, 2), view_area_2ds.shape

        assert any(
            np.array_equal(view_area_3ds[n, :2, i], view_area_2ds[n, i])
            for n in range(N)
            for i in range(5)
        ), (view_area_3ds, view_area_2ds)

        view_areas: "list[MultiPoint]" = []
        for i, view_area_2d in zip(indices, view_area_2ds):
            view_area = MultiPoint(view_area_2d.tolist())
            view_areas.append(view_area)

        # TODO: where clause should depends on query predicate
        results = database.execute(sql.SQL("""
        SELECT index
        FROM UNNEST (
            {view_areas},
            {indices}::int[]
        ) AS ViewArea(points, index)
        JOIN SegmentPolygon ON ST_Intersects(ST_ConvexHull(points), elementPolygon)
        WHERE {segment_type}
        """).format(
            view_areas=sql.Literal(view_areas),
            indices=sql.Literal(indices),
            segment_type=sql.SQL(" OR ".join(map(roadtype, self.segment_types)))
        ))

        keep = bitarray(len(payload.keep))
        keep.setall(0)
        for (index, ) in results:
            keep[index] = 1

        return keep, None


def roadtype(t: "str"):
    return f"__roadtype__{t}__"


class KeepOnlyRoadTypePredicates(BaseTransformer):
    def visit_ArrayNode(self, node: ArrayNode):
        return F.ignore_roadtype()

    def visit_CompOpNode(self, node: CompOpNode):
        return F.ignore_roadtype()

    def visit_BinOpNode(self, node: BinOpNode):
        return F.ignore_roadtype()

    def visit_BoolOpNode(self, node: BoolOpNode):
        visited = super().visit_BoolOpNode(node)
        value = (node.op == 'or')
        if any(isinstance(e, LiteralNode) and e.value == value for e in visited.exprs):
            return LiteralNode(value)
        if all(isinstance(e, LiteralNode) for e in visited.exprs):
            assert len({e.value for e in visited.exprs}) == 1
            return LiteralNode(value)
        return BoolOpNode(node.op, [e for e in visited.exprs if not isinstance(e, LiteralNode)])

    def visit_UnaryOpNode(self, node: UnaryOpNode):
        visited = super().visit_UnaryOpNode(node)

        if node.op == 'invert':
            return visited.expr

        if isinstance(visited.expr, LiteralNode):
            assert isinstance(visited.expr.value, bool), visited.expr
            return LiteralNode(not visited.expr.value)

        return visited

    def visit_LiteralNode(self, node: LiteralNode):
        if isinstance(node.value, bool):
            return node
        return F.ignore_roadtype()

    def visit_TableAttrNode(self, node: TableAttrNode):
        return F.ignore_roadtype()

    def visit_CallNode(self, node: CallNode):
        if node.fn == F.contains_all or node.fn == F.contained:
            assert (len(node.params) == 1)
            assert isinstance(node.params[0], LiteralNode), node.params[0]
            assert isinstance(node.params[0].value, str), node.params[0]
            return F.is_roadtype(node.params[0])
        return F.ignore_roadtype()

    def visit_TableNode(self, node: TableNode):
        return F.ignore_roadtype()

    def visit_ObjectTableNode(self, node: ObjectTableNode):
        return F.ignore_roadtype()

    def visit_CameraTableNode(self, node: CameraTableNode):
        return F.ignore_roadtype()

    def visit_CastNode(self, node: CastNode):
        return self(node.expr)


class PushNagationInForRoadTypePredicates(BaseTransformer):
    def visit_ArrayNode(self, node: ArrayNode):
        raise Exception('Invalid Node Type')

    def visit_CompOpNode(self, node: CompOpNode):
        raise Exception('Invalid Node Type')

    def visit_BinOpNode(self, node: BinOpNode):
        raise Exception('Invalid Node Type')

    def visit_BoolOpNode(self, node: BoolOpNode):
        assert all(isinstance(e, (BoolOpNode, CallNode, UnaryOpNode)) for e in node.exprs), node.exprs
        return node

    def visit_UnaryOpNode(self, node: UnaryOpNode):
        assert node.op == 'not'

        visited = super().visit_UnaryOpNode(node)
        expr = visited.expr

        assert isinstance(expr, (BoolOpNode, CallNode)), expr
        if isinstance(expr, BoolOpNode):
            if expr.op == 'and':
                return self(BoolOpNode('or', [~e for e in expr.exprs]))
            else:
                return self(BoolOpNode('and', [~e for e in expr.exprs]))
        else:
            assert expr.fn in [F.is_roadtype, F.is_other_roadtype, F.ignore_roadtype], expr.fn
            if expr.fn == F.is_roadtype:
                return F.is_other_roadtype(expr.params[0])
            elif expr.fn == F.is_other_roadtype:
                return F.is_roadtype(expr.params[0])
            return expr

    def visit_LiteralNode(self, node: LiteralNode):
        raise Exception('Invalid Node Type')

    def visit_TableAttrNode(self, node: TableAttrNode):
        raise Exception('Invalid Node Type')

    def visit_CallNode(self, node: CallNode):
        assert node.fn in (F.is_roadtype, F.is_other_roadtype, F.ignore_roadtype), node.fn
        return node

    def visit_TableNode(self, node: TableNode):
        raise Exception('Invalid Node Type')

    def visit_ObjectTableNode(self, node: ObjectTableNode):
        raise Exception('Invalid Node Type')

    def visit_CameraTableNode(self, node: CameraTableNode):
        raise Exception('Invalid Node Type')

    def visit_CastNode(self, node: CastNode):
        raise Exception('Invalid Node Type')


class NormalizeNagationAndFlattenRoadTypePredicates(BaseTransformer):
    def visit_ArrayNode(self, node: ArrayNode):
        raise Exception('Invalid Node Type')

    def visit_CompOpNode(self, node: CompOpNode):
        raise Exception('Invalid Node Type')

    def visit_BinOpNode(self, node: BinOpNode):
        raise Exception('Invalid Node Type')

    def visit_BoolOpNode(self, node: BoolOpNode):
        exprs = [self(e) for e in node.exprs]
        _exprs = []
        # Expand Ands
        for e in exprs:
            if isinstance(e, BoolOpNode) and e.op == node.op:
                _exprs.extend(e.exprs)
            else:
                _exprs.append(e)

        # Cleanup Ignore RoadType
        if node.op == 'and':
            if all(isinstance(e, CallNode) and e.fn == F.ignore_roadtype for e in _exprs):
                return F.ignore_roadtype()
            # Remove Ignore RoadType
            _exprs = [e for e in _exprs if not isinstance(e, CallNode) or e.fn != F.ignore_roadtype]
        else:
            if any(isinstance(e, CallNode) and e.fn == F.ignore_roadtype for e in exprs):
                return F.ignore_roadtype()

        # Boolean Absorption
        is_roadtypes = [
            e.params[0].value.lower()
            for e in _exprs
            if isinstance(e, CallNode) and e.fn == F.is_roadtype
        ]
        nested_exprs = [e for e in _exprs if isinstance(e, BoolOpNode) and e.op != node.op]
        assert len(is_roadtypes) + len(nested_exprs) == len(_exprs)

        is_roadtypes: "set[str]" = set(is_roadtypes)

        def is_absorbed(e: "CallNode | BoolOpNode"):
            if isinstance(e, CallNode):
                assert e.fn == F.is_roadtype, e.fn

                rt = e.params[0]
                assert isinstance(rt, LiteralNode), rt

                rt = rt.value.lower()
                return rt in is_roadtypes
            else:
                assert e.op == node.op, (node.op, e.op)
                all_roadtype = all(isinstance(e, CallNode) and e.fn == F.is_roadtype for e in e.exprs)
                return all_roadtype and {ee.params[0].value.lower() for ee in e.exprs}.issubset(is_roadtypes)

        nested_exprs = [
            e for e in nested_exprs
            if not any(map(is_absorbed, e.exprs))
        ]

        _exprs = [*map(F.is_roadtype, is_roadtypes), *nested_exprs]
        if len(_exprs) == 1:
            return _exprs[0]
        return BoolOpNode(node.op, _exprs)

    def visit_UnaryOpNode(self, node: UnaryOpNode):
        raise Exception('Invalid Node Type')

    def visit_LiteralNode(self, node: LiteralNode):
        raise Exception('Invalid Node Type')

    def visit_TableAttrNode(self, node: TableAttrNode):
        raise Exception('Invalid Node Type')

    def visit_CallNode(self, node: CallNode):
        assert node.fn in (F.is_roadtype, F.is_other_roadtype, F.ignore_roadtype), node.fn
        if node.fn == F.is_other_roadtype:
            rt = node.params[0]
            assert isinstance(rt, LiteralNode)

            rt: "str" = rt.value.lower()
            assert rt != 'road'
            return BoolOpNode('or', [F.is_roadtype(e) for e in OTHER_ROAD_TYPES[rt]])
        return node

    def visit_TableNode(self, node: TableNode):
        raise Exception('Invalid Node Type')

    def visit_ObjectTableNode(self, node: ObjectTableNode):
        raise Exception('Invalid Node Type')

    def visit_CameraTableNode(self, node: CameraTableNode):
        raise Exception('Invalid Node Type')

    def visit_CastNode(self, node: CastNode):
        raise Exception('Invalid Node Type')


class FindRoadTypes(Visitor["set[str]"]):
    def visit_ArrayNode(self, node: "ArrayNode") -> "set[str]":
        raise Exception('Invalid Node Type')

    def visit_CompOpNode(self, node: "CompOpNode") -> "set[str]":
        raise Exception('Invalid Node Type')

    def visit_BinOpNode(self, node: "BinOpNode") -> "set[str]":
        raise Exception('Invalid Node Type')

    def visit_BoolOpNode(self, node: "BoolOpNode") -> "set[str]":
        return set.union(*map(self, node.exprs))

    def visit_UnaryOpNode(self, node: "UnaryOpNode") -> "set[str]":
        raise Exception('Invalid Node Type')

    def visit_LiteralNode(self, node: "LiteralNode") -> "set[str]":
        raise Exception('Invalid Node Type')

    def visit_TableAttrNode(self, node: "TableAttrNode") -> "set[str]":
        raise Exception('Invalid Node Type')

    def visit_CallNode(self, node: "CallNode") -> "set[str]":
        assert node.fn == F.is_roadtype, node.fn
        return {node.params[0].value.lower()}

    def visit_TableNode(self, node: "TableNode") -> "set[str]":
        raise Exception('Invalid Node Type')

    def visit_ObjectTableNode(self, node: "ObjectTableNode") -> "set[str]":
        raise Exception('Invalid Node Type')

    def visit_CameraTableNode(self, node: "CameraTableNode") -> "set[str]":
        raise Exception('Invalid Node Type')

    def visit_CastNode(self, node: "CastNode") -> "set[str]":
        raise Exception('Invalid Node Type')


class InViewPredicate(Visitor[str]):
    def __init__(self, param_name: 'str'):
        self.param_name = param_name

    def visit_ArrayNode(self, node: "ArrayNode") -> "str":
        raise Exception('Invalid Node Type')

    def visit_CompOpNode(self, node: "CompOpNode") -> "str":
        raise Exception('Invalid Node Type')

    def visit_BinOpNode(self, node: "BinOpNode") -> "str":
        raise Exception('Invalid Node Type')

    def visit_BoolOpNode(self, node: "BoolOpNode") -> "str":
        return '(' + f' {node.op} '.join(map(self, node.exprs)) + ')'

    def visit_UnaryOpNode(self, node: "UnaryOpNode") -> "str":
        raise Exception('Invalid Node Type')

    def visit_LiteralNode(self, node: "LiteralNode") -> "str":
        raise Exception('Invalid Node Type')

    def visit_TableAttrNode(self, node: "TableAttrNode") -> "str":
        raise Exception('Invalid Node Type')

    def visit_CallNode(self, node: "CallNode") -> "str":
        assert node.fn == F.is_roadtype, node.fn

        rt = node.params[0]
        assert isinstance(rt, LiteralNode), rt

        rt: "str" = rt.value.lower()
        return f"('{rt}' in {self.param_name})"
        # return rt in self.roadtypes

    def visit_TableNode(self, node: "TableNode") -> "str":
        raise Exception('Invalid Node Type')

    def visit_ObjectTableNode(self, node: "ObjectTableNode") -> "str":
        raise Exception('Invalid Node Type')

    def visit_CameraTableNode(self, node: "CameraTableNode") -> "str":
        raise Exception('Invalid Node Type')

    def visit_CastNode(self, node: "CastNode") -> "str":
        raise Exception('Invalid Node Type')


def create_inview_predicate(
    node: "PredicateNode"
) -> "tuple[list[str], Callable[[set[str]], bool] | bool]":
    node = KeepOnlyRoadTypePredicates()(node)
    # Note True/False will either disappear from all the predicates or propagate to the top
    if isinstance(node, LiteralNode):
        assert isinstance(node.value, bool), node.value
        return [], node.value

    node = PushNagationInForRoadTypePredicates()(node)
    node = NormalizeNagationAndFlattenRoadTypePredicates()(node)
    # Note F.ignore_roadtype will either disappear from all the predicates or propagate to the top
    if isinstance(node, CallNode) and node.fn == F.ignore_roadtype:
        return [], True

    param_name = 'roadtypes'
    predicate_str = InViewPredicate(param_name)(node)
    return list(FindRoadTypes()(node)), eval(f"lambda {param_name}: {predicate_str}")
