TYPE_HUMAN = 'human'
TYPE_BICYCLE = 'bicycle'
TYPE_BUS = 'bus'
TYPE_CAR = 'car'
TYPE_MOTORCYCLE = 'motorcycle'
TYPE_TRUCK = 'truck'

OBJECT_TYPE_MAP = {
    'animal': 'animal',

    'human.pedestrian.adult': TYPE_HUMAN,
    'human.pedestrian.child': TYPE_HUMAN,
    'human.pedestrian.construction_worker': TYPE_HUMAN,
    'human.pedestrian.personal_mobility': TYPE_HUMAN,
    'human.pedestrian.police_officer': TYPE_HUMAN,
    'human.pedestrian.stroller': TYPE_HUMAN,
    'human.pedestrian.wheelchair': TYPE_HUMAN,

    'vehicle.bicycle': TYPE_BICYCLE,
    'vehicle.bus.bendy': TYPE_BUS,
    'vehicle.bus.rigid': TYPE_BUS,
    'vehicle.car': TYPE_CAR,
    'vehicle.motorcycle': TYPE_MOTORCYCLE,
    'vehicle.truck': TYPE_TRUCK,
}

def object_type(t: "str"):
    return OBJECT_TYPE_MAP.get(t, None)
