from db import Database
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
        1. For update method, we create two nodes: the first node will write to the db, the second node will retrieve from the db
        2. For the write node, never double write. (so we use done flag)
        ... -> [write] -> [retrive] -> ...
        """
        node1 = self._insert_camera(*args, **kwargs)
        node2 = node1._retrieve_camera(world_id = node1.world_id)
        return node2

    def predicate(self, *args, **kwargs):
        new_node = self._create_new_world_and_link()
        new_node.fn = self.db.nest_from
        new_node.args, new_node.kwargs = args, kwargs
        return new_node

    def _insert_camera(self, *args, **kwargs):
        new_node = self._create_new_world_and_link()
        new_node.fn = self.db.insert_cam
        new_node.args, new_node.kwargs = args, {**kwargs, "world_id": new_node.world_id}
        return new_node

    def _retrieve_camera(self, *args, **kwargs):
        new_node = self._create_new_world_and_link()
        new_node.fn = self.db.concat_with
        new_node.args, new_node.kwargs = args, kwargs
        return new_node

    def get_camera(self, *args, **kwargs):
        new_node = self._create_new_world_and_link()
        new_node.fn = self.db.execute_get_query
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
                query = node._execute(query = query)

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
    
    # w1 = World()

    # w2 = w1.add_camera(cam_id = "1", cam_size = 5)

    # w3 = w2.predicate(condition = "query.size < 4")

    # w4 = w3.add_camera(cam_id = "2", cam_size = 3)

    # w5 = w4.predicate(condition = "query.size < 6")

    # res = w5.get_camera()

    # print(list(res))

    """
    w1 - w2 - w3 - w311 - w312  
                 - w321 - w322
    """
    w1 = World()

    w2 = w1.add_camera(cam_id = "c2", cam_size = 1)

    w3 = w2.add_camera(cam_id = "c3", cam_size = 1)

    w311 = w3.add_camera(cam_id = "c311", cam_size = 1)

    w312 = w311.add_camera(cam_id = "c312", cam_size = 1)

    w313 = w312.add_camera(cam_id = "c313", cam_size = 1)

    w321 = w3.add_camera(cam_id = "c321", cam_size = 1)

    w322 = w321.add_camera(cam_id = "c322", cam_size = 1)

    w323 = w321.add_camera(cam_id = "c323", cam_size = 1)

    res3 = w3.get_camera()
    print(list(res3)) # [('c3', 1, '55145a19-db3a-4a4d-af1c-00a266fe1964'), ('c2', 1, '2a7a17b5-783e-4881-9326-c878565cfd40')]

    res322 = w322.get_camera()
    print(list(res322)) # [('c2', 1, '2a7a17b5-783e-4881-9326-c878565cfd40'), ('c3', 1, '55145a19-db3a-4a4d-af1c-00a266fe1964'), ('c321', 1, '7c40fce0-5254-44ea-9692-2e735a263aa3'), ('c322', 1, 'a46270e2-8893-4945-819a-9259423a7b5d')]

    res312 = w312.get_camera()
    print(list(res312)) # [('c2', 1, '2a7a17b5-783e-4881-9326-c878565cfd40'), ('c3', 1, '55145a19-db3a-4a4d-af1c-00a266fe1964'), ('c311', 1, 'b257beab-d4d1-40e9-8744-e574b7060285'), ('c312', 1, '59b9f8ee-643c-448b-bd01-bab86e8dedcd')]
