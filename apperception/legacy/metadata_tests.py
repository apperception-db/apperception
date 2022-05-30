import unittest

import psycopg2

from apperception.legacy.metadata_context import (MetadataContext, geometry,
                                                  primarykey, time)
from apperception.legacy.metadata_context_executor import \
    MetadataContextExecutor

test_context = MetadataContext()

conn = psycopg2.connect(
    database="mobilitydb", user="docker", password="docker", host="localhost", port=5432
)

test_executor = MetadataContextExecutor(conn)
# test_executor.connect_db(user="postgres", password="postgres", database_name="postgres")
# Test simple queries using Context class


class TestStringMethods(unittest.TestCase):
    def test_commands(self):
        test_executor.context(test_context.selectkey())
        print(test_executor.execute())
        print("------------------------------------")

        test_executor.context(test_context.get_trajectory())
        print(test_executor.execute())
        print("------------------------------------")

        test_executor.context(
            test_context.get_geo().interval("0001-01-01 00:00:00", "9999-12-31 23:59:59.999999")
        )
        print(test_executor.execute())
        print("------------------------------------")

        test_executor.context(test_context.get_geo())
        print(test_executor.execute())
        print("------------------------------------")

        test_executor.context(test_context.get_time())
        print(test_executor.execute())
        print("------------------------------------")

        test_executor.context(test_context.get_speed())
        print(test_executor.execute())
        print("------------------------------------")

        test_executor.context(test_context.get_distance())
        print(test_executor.execute())
        print("------------------------------------")

        test_executor.context(test_context.get_columns(primarykey, geometry, time))
        print("###### bboxes and times are:    ", test_executor.execute())
        print("------------------------------------")

        # test_executor.context(test_context.count(MetadataContext.selectkey))
        # print(test_executor.execute())
        # print("------------------------------------")

    def test_usecases(self):
        # test_executor.context(test_context.predicate(lambda  obj:obj.object_id == "Item_1").get_geo())
        # print(test_executor.execute())
        # print("------------------------------------")

        # This query could be confusing since the user may understand it as getting the trajectory of the objects when they are at the intersection
        # but the trajectory is actually an attribute, so it's always the entire trajectory
        # If the user really wants to get a certain period of trajectory they have to filter out the timestamps
        volume = "stbox 'STBOX Z((1.81788543, 2.17411856, 0),(2.79369985, 3.51919659, 2))'"
        filtered_world = test_context.predicate(lambda obj: obj.object_type == "car").predicate(
            lambda obj: obj.location in volume, {"volume": volume}
        )
        trajectory = filtered_world.get_trajectory(distinct=True)
        test_executor.context(trajectory)
        print(test_executor.execute())
        print("------------------------------------")

        # to get the video over the entire trajectory(amber case)
        test_executor.context(filtered_world.selectkey(distinct=True))
        filtered_ids = test_executor.execute()
        print("filtered_IDS are *****:", filtered_ids)

        id_array = [filtered_id[0] for filtered_id in filtered_ids]
        entire_video = test_context.predicate(
            lambda obj: obj.object_id in id_array, {"id_array": id_array}
        ).get_columns(primarykey, geometry, time)
        test_executor.context(entire_video)
        print(test_executor.execute())
        print("------------------------------------")

        # test_executor.context(test_context.predicate(lambda obj:obj.color == "red").group(get_time).predicate(lambda obj:count(obj) >= 3).get_time())
        # print(test_executor.execute())
        # print("------------------------------------")

    # def test_table_join(self):
    #     ### Inner Join
    #     new_meta_context = test_context.selectkey().get_distance().get_speed().view().join(metadata_view) ### create a temporary view without reference
    #     test_executor.context(new_meta_context.predicate(lambda obj:obj.object_type == 'car'))
    #     car_newmeta = test_executor.execute()
    #     print(car_newmeta)
    #     print("------------------------------------")

    #     test_executor.context(new_meta_context.predicate(lambda obj:obj.object_type == 'car').view(view_name="car_view"))
    #     car_newmeta_view = test_executor.execute()
    #     print(car_newmeta_view) ### this should be the same result as previous execution
    #     print("------------------------------------")

    #     ### Query from new view
    #     test_executor.context(test_context.view(use_view = car_newmeta_view).selectkey().get_trajectory().get_speed())
    #     print(test_executor.execute())
    #     print("------------------------------------")

    # def test_mix(self):
    #     stbox = "stbox \'STBOX Z((1.81788543, 2.17411856, 0),(2.79369985, 3.51919659, 2))\'"
    #     proposal_context = test_context.get_trajectory().predicate(lambda obj:obj.object_type == 'car').predicate(lambda obj:obj.location in volume, {"volume":stbox})
    #     test_executor.context(proposal_context)
    #     print(test_executor.execute())
    #     print("------------------------------------")

    #     test_executor.context(test_context.count(key=MetadataContext.selectkey).predicate(lambda obj: obj.color == "red").group(lambda obj: obj.color))
    #     print(test_executor.execute())
    #     print("------------------------------------")

    #     test_executor.context(test_context.get_time().predicate(lambda obj:obj.color == "red" and obj.location in volume and count(obj.id), {"volume":stbox}).group(lambda obj: obj.color))
    #     print(test_executor.execute())
    #     print("------------------------------------")
    # def test_usecases(self):
    #     TODO: Define use cases here


if __name__ == "__main__":
    unittest.main()
