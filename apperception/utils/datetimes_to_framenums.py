
from datetime import datetime
from typing import List


def datetimes_to_framenums(start_time: datetime, datetimes: List[datetime]) -> List[int]:
    return [int((t.replace(tzinfo=None) - start_time).total_seconds()) for t in datetimes]