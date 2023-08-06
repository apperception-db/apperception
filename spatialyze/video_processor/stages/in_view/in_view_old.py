from typing import Dict, List, Optional, Tuple

from bitarray import bitarray

from spatialyze.database import database

from ...camera_config import Float3
from ...payload import Payload
from ..stage import Stage


class InViewOld(Stage):
    def __init__(self, distance: float, segment_type: str, min_distance=False) -> None:
        super().__init__()
        self.distance = distance
        self.segment_type = segment_type
        self.min_distance = min_distance

    def _run(self, payload: "Payload") -> "Tuple[Optional[bitarray], Optional[Dict[str, list]]]":
        keep = bitarray(payload.keep)
        translations: "List[Float3]" = []
        headings: "List[float]" = []
        indices: "List[int]" = []
        for i, f in enumerate(payload.video):
            if keep[i]:
                translations.append(f.ego_translation)
                headings.append(f.ego_heading)
                indices.append(i)

        translations_str = ",\n".join(map(_tuple_to_point, translations))
        headings_str = ",\n".join(map(str, headings))
        results = database.execute(
            f"""
            SELECT
                {f"minDistance(t, '{self.segment_type}'::text) < {self.distance} AND" if self.min_distance else ""}
                inView('intersection', h, t, {self.distance}, 35)
            FROM
                UNNEST(
                    ARRAY[{translations_str}],
                    ARRAY[{headings_str}]
                ) AS ego(t, h)"""
        )

        for i, (r,) in enumerate(results):
            keep[indices[i]] = r

        return keep, None


def _tuple_to_point(t: "Float3"):
    return f"'POINT Z ({' '.join(map(str, t))})'::geometry"
