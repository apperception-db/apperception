import json
import os
import pickle
import pytest
import numpy as np

from spatialyze.predicate import *
from spatialyze.utils import F

from optimized_ingestion.stages.in_view.in_view import FindRoadTypes, InViewPredicate, KeepOnlyRoadTypePredicates, NormalizeInversionAndFlattenRoadTypePredicates, PushInversionInForRoadTypePredicates, InView
from optimized_ingestion.pipeline import Pipeline
from optimized_ingestion.payload import Payload
from optimized_ingestion.video import Video
from optimized_ingestion.camera_config import camera_config

from optimized_ingestion.stages.decode_frame.decode_frame import DecodeFrame
from optimized_ingestion.stages.detection_2d.yolo_detection import YoloDetection

# Test Strategies
# - Real use case -- simple predicates from the query
# - 2 format + add true/false + add ignore_roadtype
#   - x AND ~y AND (z OR ~ w)
#   - x OR ~y OR (z AND ~ w)


o = objects[0]
o1 = objects[1]
o2 = objects[2]
c = camera
c0 = cameras[0]
gen = GenSqlVisitor()
RT = '__ROADTYPES__'

@pytest.mark.parametrize("fn, sqls", [
    # Propagate Boolean
    (o.type & False, ['False']),
    (o.type | True, ['True']),

    (o.type | (~(o.a & False)), ['True']),
    (o.type & (~(o.a | True)), ['False']),

    (o.type & (~((o.a | True) & True)), ['False']),
    (o.type | (~((o.a & False) | False)), ['True']),

    # General
    (o.type & True, ['ignore_roadtype()', None, None]),
    (o.type & True & F.contained(o.traj, 'intersection'), [
        '(ignore_roadtype() AND is_roadtype(intersection))',
        None,
        'is_roadtype(intersection)',
        f"('intersection' in {RT})",
        {'intersection'}
    ]),

    # Real Queries
    ((((o1.type == 'car') | (o1.type == 'truck')) &
        F.angle_between(F.facing_relative(c.ego, F.road_direction(c.ego)), -15, 15) &
        (F.distance(c.ego, o1.trans@c.time) < 50) &
        F.contains_all('intersection', [o1.trans]@c.time) &
        F.angle_between(F.facing_relative(o1.trans@c.time, c.ego), 135, 225) &
        (F.min_distance(c.ego, 'intersection') < 10)), [
        '((ignore_roadtype() OR ignore_roadtype()) AND ignore_roadtype() AND ignore_roadtype() AND is_roadtype(intersection) AND ignore_roadtype() AND ignore_roadtype())',
        None,
        'is_roadtype(intersection)',
        f"('intersection' in {RT})",
        {'intersection'}
    ]),
    (((((o1.type == 'car') | (o1.type == 'truck')) &
        F.angle_between(F.facing_relative(c.ego, F.road_direction(c.ego)), -15, 15) &
        (F.distance(c.ego, o1.trans@c.time) < 50) &
        F.angle_between(F.facing_relative(o1.trans@c.time, c.ego), 135, 225) &
        (F.min_distance(c.ego, 'intersection') < 10)) |
        F.contains_all('intersection', [o1.trans]@c.time)), [
        '(((ignore_roadtype() OR ignore_roadtype()) AND ignore_roadtype() AND ignore_roadtype() AND ignore_roadtype() AND ignore_roadtype()) OR is_roadtype(intersection))',
        None,
        'ignore_roadtype()',
    ]),
    ((((o1.type == 'car') | (o1.type == 'truck')) &
        F.contains_all('intersection', [o1.trans]@c.time) &
        F.contains_all('lanesection', [o1.trans]@c.time) &
        (F.min_distance(c.ego, 'intersection') < 10)), [
        '((ignore_roadtype() OR ignore_roadtype()) AND is_roadtype(intersection) AND is_roadtype(lanesection) AND ignore_roadtype())',
        None,
        '(is_roadtype(intersection) AND is_roadtype(lanesection))',
        f"(('intersection' in {RT}) and ('lanesection' in {RT}))",
        {'intersection', 'lanesection'}
    ]),
    ((((o1.type == 'car') | (o1.type == 'truck')) &
        F.contains_all('intersection', [o1.trans]@c.time) &
        ~F.contains_all('lanesection', [o1.trans]@c.time) &
        (F.min_distance(c.ego, 'intersection') < 10)), [
        '((ignore_roadtype() OR ignore_roadtype()) AND is_roadtype(intersection) AND (NOT is_roadtype(lanesection)) AND ignore_roadtype())',
        '((ignore_roadtype() OR ignore_roadtype()) AND is_roadtype(intersection) AND is_other_roadtype(lanesection) AND ignore_roadtype())',
        'is_roadtype(intersection)',
        f"('intersection' in {RT})",
        {'intersection'}
    ]),
])
def test_predicates(fn, sqls):
    node = KeepOnlyRoadTypePredicates()(fn)
    assert gen(node) == sqls[0], node
    
    if len(sqls) > 1:
        sql = sqls[1]
        node1 = PushInversionInForRoadTypePredicates()(node)
        assert gen(node1) == (gen(node) if sql is None else sql), node1
    
    if len(sqls) > 2:
        sql = sqls[2]
        node2 = NormalizeInversionAndFlattenRoadTypePredicates()(node1)
        assert gen(node2) == (gen(node1) if sql is None else sql), node2
    
    if len(sqls) > 3:
        assert isinstance(sqls[3], str), sqls[3]
        assert isinstance(sqls[4], set), sqls[4]

        predicate_str = InViewPredicate(RT)(node2)
        assert predicate_str == sqls[3], predicate_str

        roadtypes = FindRoadTypes()(node2)
        assert roadtypes == sqls[4], roadtypes


# TODO: add these predicates

# @pytest.mark.parametrize("fn, sql", [
#     (o.c1 + c.c1, "true"),
#     (o.c1 == c.c1, "true"),
#     (o.c1 < c.c1, "true"),
#     (o.c1 != c.c1, "true"),

#     (lit(3), "3"),
#     (lit('test', False), "test"),

#     (c0.c1, "true"),
#     (cast(c0.c1, 'real'), "true"),

#     (-o.c1, "(-t0.c1)"),
#     (~o.c1, "(NOT t0.c1)"),
#     (~F.contained('intersection', o.c1), "(NOT SegmentPolygon.__RoadType__intersection__)"),
#     (~F.contains_all('intersection', o.c1), "(NOT SegmentPolygon.__RoadType__intersection__)"),
#     (o.c1 & ~F.contained('intersection', o.c1) & F.contained('intersection', o.c1), "(true AND true AND SegmentPolygon.__RoadType__intersection__)"),
#     (o.c1 | ~F.contained('intersection', o.c1) | F.contained('intersection', o.c1), "(true OR true OR SegmentPolygon.__RoadType__intersection__)"),
#     (o.c1 @ c.timestamp, "valueAtTimestamp(t0.c1,timestamp)"),
#     (c.timestamp @ 1, "valueAtTimestamp(timestamp,1)"),
#     ([o.c1, o.c2] @ c.timestamp, "ARRAY[valueAtTimestamp(t0.c1,timestamp),valueAtTimestamp(t0.c2,timestamp)]"),
#     (o.bbox @ c.timestamp, "objectBBox(t0.itemId,timestamp)"),
# ])
# def test_simple_ops(fn, sql):

#     assert gen(normalize(fn)) == sql


# @pytest.mark.parametrize("fn, sql", [
#     ((o.c1 + c.c1) - c.c2 + o.c2 * c.c3 / o.c3, "(((t0.c1+c1)-c2)+((t0.c2*c3)/t0.c3))"),
#     ((o.c1 == c.c1) & ((o.c2 < c.c2) | (o.c3 == c.c3)), "((t0.c1=c1) AND ((t0.c2<c2) OR (t0.c3=c3)))"),
# ])
# def test_nested(fn, sql):
#     assert gen(normalize(fn)) == sql


# @pytest.mark.parametrize("fn, tables, camera", [
#     (o.c1 & o1.c2 & c.c3, {0, 1}, True),
#     ((o.c1 + c.c1) - c.c2 + o.c2 * c.c3 / o.c3, {0}, True),
#     ((o.c1) + o1.c2 / o.c3, {0, 1}, False),
#     ((o.c1) + c.c2 / o.c3, {0}, True),
# ])
# def test_find_all_tables(fn, tables, camera):
#     assert FindAllTablesVisitor()(normalize(fn)) == (tables, camera)

OUTPUT_DIR = './data/pipeline/test-results'
VIDEO_DIR =  './data/pipeline/videos'

def test_detection_2d():
    files = os.listdir(VIDEO_DIR)

    with open(os.path.join(VIDEO_DIR, 'frames.pkl'), 'rb') as f:
        videos = pickle.load(f)
    
    for distance in [10, 20, 30, 40, 50]:
        pipeline1 = Pipeline([InView(distance, roadtypes='intersection')])
        pipeline2 = Pipeline([InView(distance, predicate=(
            ((o1.type == 'car') | (o1.type == 'truck')) &
            F.contains_all('intersection', [o1.trans]@c.time) &
            ~F.contains_all('lanesection', [o1.trans]@c.time) &
            (F.min_distance(c.ego, 'intersection') < 10))
        )])

        for name, video in videos.items():
            if video['filename'] not in files:
                continue
            
            frames = Video(
                os.path.join(VIDEO_DIR, video["filename"]),
                [camera_config(*f, 0) for f in video["frames"]],
            )

            output1 = pipeline1.run(Payload(frames))
            output2 = pipeline2.run(Payload(frames))

            assert output1.keep == output2.keep, (name, output1.keep, output2.keep)
