from apperception.predicate import (BinOpNode, CallNode, CompOpNode,
                                    LiteralNode, PredicateNode, TableAttrNode,
                                    Visitor)
from optimized_ingestion.stages.detection_2d.object_type_filter import \
    ObjectTypeFilter
from optimized_ingestion.stages.detection_2d.yolo_detection import \
    YoloDetection
from optimized_ingestion.stages.in_view import InView


def in_view(pipeline, param):
    pipeline.stages.insert(0, InView(**param))


def object_type(pipeline, param):
    # pass
    for i in range(len(pipeline.stages)):
        if isinstance(pipeline.stages[i], YoloDetection):
            if isinstance(pipeline.stages[i + 1], ObjectTypeFilter):
                pipeline.stages[i + 1].add_type(param)
            else:
                assert isinstance(param, str)
                pipeline.stages.insert(i + 1, ObjectTypeFilter([param]))


def road_type(pipeline, param):
    ### TODO(fge): reenable pruning after fixing detection info
    pass
    # for s in pipeline.stages:
    #     if isinstance(s, DetectionEstimation):
    #         s.add_filter(lambda x: x.road_type == param)


def distance_to_ego(pipeline, param):
    pass
    # for s in pipeline.stages:
    #     if isinstance(s, DetectionEstimation):
    #         s.add_filter(lambda x: compute_distance(
    #             x.car_loc3d, x.ego_config.ego_translation) < param)


ALL_MAPPING_RULES = {
    'in_view': {'condition': lambda x: (isinstance(x, CompOpNode) and
                                        isinstance(x.left, CallNode)
                                        and isinstance(x.right, LiteralNode)
                                        and x.left._fn[0].__name__ == 'fn'
                                        and isinstance(x.left.params[0], TableAttrNode)
                                        and x.left.params[0].name == 'egoTranslation'
                                        and isinstance(x.left.params[1], LiteralNode)),
                'param': lambda x: dict(segment_type=x.left.params[1].value, distance=x.right.value),
                'pipeline': in_view},
    'object_type': {'condition': lambda x: (isinstance(x, CallNode)
                                            and x._fn[0].__name__ == 'like'
                                            and x.params[0].name == 'objectType'),
                    'param': lambda x: x.params[1].value,
                    'pipeline': object_type},
    'road_type': {'condition': lambda x: (isinstance(x, CallNode)
                                          and x._fn[0].__name__ == 'contains_all'),
                  'param': lambda x: x.params[0].value,
                  'pipeline': road_type},
    'distance_to_ego': {'condition': lambda x: (isinstance(x, CompOpNode) and
                                                isinstance(x.left, CallNode)
                                                and isinstance(x.right, LiteralNode)
                                                and x.left._fn[0].__name__ == 'fn'
                                                and isinstance(x.left.params[0], TableAttrNode)
                                                and x.left.params[0].name == 'egoTranslation'
                                                and isinstance(x.left.params[1], BinOpNode)),
                        'param': lambda x: x.right.value,
                        'pipeline': distance_to_ego}
}


def pipeline_rule(pipeline, node):
    for key, rule in ALL_MAPPING_RULES.items():
        if rule['condition'](node):
            param = rule['param'](node)
            rule['pipeline'](pipeline, param)


class PipelineConstructor(Visitor[PredicateNode]):

    def add_pipeline(self, pipeline):
        self.pipeline = pipeline
        return self

    def visit_CompOpNode(self, node: "CompOpNode"):
        assert self.pipeline
        pipeline_rule(self.pipeline, node)
        self(node.left)
        self(node.right)

    def visit_CallNode(self, node: "CallNode"):
        assert self.pipeline
        pipeline_rule(self.pipeline, node)
        for p in node.params:
            self(p)
