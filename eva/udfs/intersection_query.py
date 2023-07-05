from evadb.udfs.abstract.abstract_udf import AbstractUDF
from evadb.udfs.decorators.decorators import forward, setup


class IntersectionQuery(AbstractUDF):
    @setup(cacheable=True, udf_type="object_detection", batchable=True)
    def setup(self):
        pass
    
    @forward(
        input_signatures=[],
        output_signatures=[],
    )
    def forward(self, frames):
        metadata = self.md.eval_all(frames)

        return metadata

    def name(self):
        return "IntersectionQuery"