from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from bitarray import bitarray

from apperception.database import database

from .stage import Stage

if TYPE_CHECKING:
    from ..camera_config import Float3
    from ..payload import Payload


class InView(Stage):
    def __init__(self, distance: float, segment_type: str) -> None:
        self.distance = distance
        self.segment_type = segment_type

    def __call__(self, payload: "Payload") -> "Tuple[Optional[bitarray], Optional[Dict[str, list]]]":
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
        results = database._execute_query(
            f"""
            SELECT
                minDistance(t, '{self.segment_type}') < {self.distance} AND
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
    return f"'POINT Z ({' '.join(map(str, t))})'"
