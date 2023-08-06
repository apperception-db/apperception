from functools import reduce
from typing import List, Tuple


def partition_by_cpus(num_cpus: int, num_elements: int):
    q, mod = divmod(num_elements, num_cpus)
    elements_per_cpu = [q + (i < mod) for i in range(num_cpus)]
    return reduce(_reducer, elements_per_cpu, (0, []))[1]


def _reducer(state: "Tuple[int, List[Tuple[int, int]]]", elements: int):
    start, arr = state
    end = start + elements
    return (end, arr + [(start, end)])
