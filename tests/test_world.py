from apperception.world import empty_world, World


COND1 = "F.like(o.object_type, 'vehicle%')"
COND2 = "F.like(o.object_type, 'human%')"


def sort_key(world: "World"):
    keys = [w[0] for w in world.get_traj_key()]
    return sorted(keys)


def test_union():
    world = empty_world('world')
    w1 = world.filter(f"lambda o: {COND1} or {COND2}")
    w2 = world.filter(f"lambda o: {COND1}") | world.filter(f"lambda o: {COND2}")
    assert sort_key(w1) == sort_key(w2)


def test_intersect():
    world = empty_world('world')
    w1 = world.filter(f"lambda o: {COND1} and {COND2}")
    w2 = world.filter(f"lambda o: {COND1}") & world.filter(f"lambda o: {COND2}")
    w3 = world.filter(f"lambda o: {COND1}").filter(f"lambda o: {COND2}")
    w4 = world.filter("lambda o: False")
    assert sort_key(w1) == sort_key(w2)
    assert sort_key(w1) == sort_key(w3)
    assert sort_key(w1) == sort_key(w4)


def test_exclude():
    world = empty_world('world')
    w1 = world.filter(f"lambda o: {COND1} and (not {COND2})")
    w2 = world.filter(f"lambda o: {COND1}") - world.filter(f"lambda o: {COND2}")
    assert sort_key(w1) == sort_key(w2)


def test_sym_diff():
    world = empty_world('world')
    w1 = world.filter(f"lambda o: {COND1} != {COND2}")
    w2 = world.filter(f"lambda o: {COND1}") ^ world.filter(f"lambda o: {COND2}")
    assert sort_key(w1) == sort_key(w2)
