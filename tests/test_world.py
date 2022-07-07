from typing import List
from apperception.world import empty_world, World


COND1 = "F.like(o.object_type, 'vehicle%')"
COND2 = "F.like(o.object_type, 'human%')"
COND3 = "o.cameraid = 'scene-0816'"


def sort_keys(*worlds: "World") -> List[List[str]]:
    return [sort_key(w) for w in worlds]


def sort_key(world: "World") -> List[str]:
    keys = [w[0] for w in world.get_traj_key()]
    return sorted(keys)


def all_equal(*keys: List[str]):
    f = keys[0]
    for key in keys[1:]:
        assert f == key


def test_union():
    world = empty_world('world')
    w1 = world.filter(f"lambda o: {COND1} or {COND2}")
    w2 = world.filter(f"lambda o: {COND1}") | world.filter(f"lambda o: {COND2}")
    k1, k2 = sort_keys(w1, w2)
    all_equal(k1, k2)

    w1 = world.filter(f"lambda o: {COND1} or {COND3}")
    w2 = world.filter(f"lambda o: {COND1}") | world.filter(f"lambda o: {COND3}")
    k1, k2 = sort_keys(w1, w2)
    all_equal(k1, k2)


def test_intersect():
    world = empty_world('world')
    w1 = world.filter(f"lambda o: {COND1} and {COND2}")
    w2 = world.filter(f"lambda o: {COND1}") & world.filter(f"lambda o: {COND2}")
    w3 = world.filter(f"lambda o: {COND1}").filter(f"lambda o: {COND2}")
    w4 = world.filter("lambda o: False")
    k1, k2, k3, k4 = sort_keys(w1, w2, w3, w4)
    all_equal(k1, k2, k3, k4)

    w1 = world.filter(f"lambda o: {COND1} and {COND3}")
    w2 = world.filter(f"lambda o: {COND1}") & world.filter(f"lambda o: {COND3}")
    w3 = world.filter(f"lambda o: {COND1}").filter(f"lambda o: {COND3}")
    k1, k2, k3 = sort_keys(w1, w2, w3)
    all_equal(k1, k2, k3)


def test_exclude():
    world = empty_world('world')
    w1 = world.filter(f"lambda o: {COND1} and (not {COND2})")
    w2 = world.filter(f"lambda o: {COND1}") - world.filter(f"lambda o: {COND2}")
    k1, k2 = sort_keys(w1, w2)
    all_equal(k1, k2)

    w1 = world.filter(f"lambda o: {COND1} and (not {COND3})")
    w2 = world.filter(f"lambda o: {COND1}") - world.filter(f"lambda o: {COND3}")
    k1, k2 = sort_keys(w1, w2)
    all_equal(k1, k2)


def test_sym_diff():
    world = empty_world('world')
    w1 = world.filter(f"lambda o: {COND1} != {COND2}")
    w2 = world.filter(f"lambda o: {COND1}") ^ world.filter(f"lambda o: {COND2}")
    k1, k2 = sort_keys(w1, w2)
    all_equal(k1, k2)

    w1 = world.filter(f"lambda o: {COND1} != {COND3}")
    w2 = world.filter(f"lambda o: {COND1}") ^ world.filter(f"lambda o: {COND3}")
    k1, k2 = sort_keys(w1, w2)
    all_equal(k1, k2)
