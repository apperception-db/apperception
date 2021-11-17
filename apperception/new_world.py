from new_db import Database
from new_util import create_camera
import uuid

class World:
    # all worlds share a db instance
    db = Database()

    def __init__(self):
        self.fn = None
        self.args = None
        self.kwargs = None
        self.parent = None
        self.done = False
        self.world_id = str(uuid.uuid4())

    def add_camera(self, *args, **kwargs):
        """
        1. For update method, we create two nodes: the first node will write to the db, and the second node will retrieve from the db
        2. For the write node, never double write. (so we use done flag)
        ... -> [write] -> [retrive] -> ...
        """
        node1 = self._insert_camera(*args, **kwargs)
        node2 = node1._retrieve_camera(world_id = node1.world_id)
        return node2

    def predicate(self, *args, **kwargs):
        new_node = self._create_new_world_and_link()
        new_node.fn = self.db.filter_cam
        new_node.args, new_node.kwargs = args, kwargs
        return new_node

    def _insert_camera(self, *args, **kwargs):
        new_node = self._create_new_world_and_link()
        new_node.fn = self.db.insert_cam
        new_node.args, new_node.kwargs = args, {**kwargs, "world_id": new_node.world_id}
        return new_node

    def _retrieve_camera(self, *args, **kwargs):
        new_node = self._create_new_world_and_link()
        new_node.fn = self.db.retrieve_cam
        new_node.args, new_node.kwargs = args, kwargs
        return new_node

    def get_camera(self, *args, **kwargs):
        new_node = self._create_new_world_and_link()
        new_node.fn = self.db.get_cam
        new_node.args, new_node.kwargs = args, kwargs
        return new_node._execute_from_root()

    def _create_new_world_and_link(self):
        new_world = World()
        new_world.parent = self
        return new_world

    def _execute_from_root(self):
        nodes = []
        curr = self
        res = None
        query = ""

        # collect all the nodes til the root
        while curr:
            nodes.append(curr)
            curr = curr.parent

        # execute the nodes from the root
        for node in nodes[::-1]:
            # root
            if node.fn == None:
                continue
            # treat update method differently
            elif node.fn == self.db.insert_cam:
                if node.done == False:
                    node._execute()
                    node.done = True
            else:
                print(query)
                query = node._execute(query=query)

        res = query
        return res

    def _execute(self, *args, **kwargs):
        # print("executing fn = {}, with args = {} and kwargs = {}".format(self.fn, self.args, self.kwargs))
        return self.fn(*self.args, *args, **self.kwargs, **kwargs)

    def _print_til_root(self):
        curr = self
        while curr:
            print(curr)
            curr = curr.parent

    def __str__(self):
        return "fn={}\nargs={}\nkwargs={}\ndone={}\nworld_id={}\n".format(self.fn, self.args, self.kwargs, self.done, self.world_id)

if __name__ == "__main__":

    w1 = World()

    c2 = create_camera(cam_id="cam2", fov=60)

    w2t = w1.add_camera(camera_node=c2)

    w2 = w2t.predicate(condition="query.fov < 310")
    #
    c3 = create_camera(cam_id="cam3", fov=120)

    w3 = w2.add_camera(camera_node=c3)

    w4 = w3.predicate(condition="query.fov < 300")

    res = w4.get_camera()

    print(res)
