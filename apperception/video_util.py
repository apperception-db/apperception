from typing import Any, Dict


def video_data_to_tasm(video_file, metadata_id, t):
    t.store(video_file, metadata_id)


def metadata_to_tasm(formatted_result: Dict[str, Any], metadata_id, t):
    import tasm

    metadata_info = []

    def bound_width(x):
        return min(max(0, x), 3840)

    def bound_height(y):
        return min(max(0, y), 2160)

    for obj, info in formatted_result.items():
        object_type = info.object_type
        for bbox, frame in zip(info.bboxes, info.tracked_cnt):
            x1 = bound_width(bbox.x1)
            y1 = bound_height(bbox.y1)
            x2 = bound_width(bbox.x2)
            y2 = bound_height(bbox.y2)
            if frame < 0 or x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0:
                import pdb

                pdb.set_trace()
            metadata_info.append(tasm.MetadataInfo(metadata_id, object_type, frame, x1, y1, x2, y2))
            metadata_info.append(tasm.MetadataInfo(metadata_id, obj, frame, x1, y1, x2, y2))

    t.add_bulk_metadata(metadata_info)


def create_or_insert_world_table(conn, name, units):
    # Creating a cursor object using the cursor() method
    cursor = conn.cursor()
    """
	Create and Populate A world table with the given world object.
	"""
    # Doping Worlds table if already exists.
    cursor.execute("DROP TABLE IF EXISTS Worlds;")
    # Creating table with the first world
    sql = """CREATE TABLE IF NOT EXISTS Worlds(
	worldId TEXT PRIMARY KEY,
	units TEXT
	);"""
    cursor.execute(sql)
    print("Worlds Table created successfully........")
    insert_world(conn, name, units)
    return sql


# Helper function to insert the world


def insert_world(conn, name, units):
    # Creating a cursor object using the cursor() method
    cursor = conn.cursor()
    cursor.execute(
        """INSERT INTO Worlds (worldId, units) """ + """VALUES (\'%s\',  \'%s\');""" % (name, units)
    )
    print("New world inserted successfully........")
    # Insert the existing cameras of the current world into the camera table
    conn.commit()
