from apperception.world import empty_world


COND1 = "F.like(o.object_type, 'vehicle%')"
COND2 = "F.like(o.object_type, 'human%')"


def test_union():
    world = empty_world('world')
    k1 = world.filter(f"lambda o: {COND1} or {COND2}").get_traj_key()
    k2 = world.filter(f"lambda o: {COND1}") | world.filter(f"lambda o: {COND2}").get_traj_key()
    assert k1 == k2


def test_intersect():
    world = empty_world('world')
    k1 = world.filter(f"lambda o: {COND1} and {COND2}").get_traj_key()
    k2 = world.filter(f"lambda o: {COND1}") & world.filter(f"lambda o: {COND2})").get_traj_key()
    k3 = world.filter(f"lambda o: {COND1}").filter(f"lambda o: {COND2})").get_traj_key()
    k4 = world.get_traj_key()
    assert k1 == k2
    assert k2 == k3
    assert k3 == k4


def test_exclude():
    world = empty_world('world')
    k1 = world.filter(f"lambda o: {COND1} and (not {COND2})").get_traj_key()
    k2 = world.filter(f"lambda o: {COND1}") - world.filter(f"lambda o: {COND2})").get_traj_key()
    assert k1 == k2


def test_sym_diff():
    world = empty_world('world')
    k1 = world.filter(f"lambda o: {COND1} != {COND2}").get_traj_key()
    k2 = world.filter(f"lambda o: {COND1}") ^ world.filter(f"lambda o: {COND2})").get_traj_key()
    assert k1 == k2
