import ast
import psycopg2
import numpy as np
import datetime 
import cv2
from object_tracker import yolov4_deepsort_video_track
from video_util import *

# Create a camera table
def create_or_insert_scenic_camera_table(conn, world_name, camera):
	#Creating a cursor object using the cursor() method
	cursor = conn.cursor()
	'''
	Create and Populate A camera table with the given camera object.
	'''
	### TODO: Modify the following codes to for scenic cameras
	#Doping Cameras table if already exists.
	cursor.execute("DROP TABLE IF EXISTS Cameras")
	#Creating table with the first camera
	sql = '''CREATE TABLE IF NOT EXISTS Scenic_Cameras(
	cameraId TEXT,
	worldId TEXT,
	ratio real,
	origin geometry,
	focalpoints geometry,
	fov INTEGER,
	skev_factor real
	);'''
	cursor.execute(sql)
	print("Camera Table created successfully........")
	insert_scenic_camera(conn, world_name, camera)
	return sql

# Helper function to insert the camera
def insert_scenic_camera(conn, world_name, camera_node):
	#Creating a cursor object using the cursor() method
	cursor = conn.cursor()
	lens = camera_node.lens
	focal_x = str(lens.focal_x)
	focal_y = str(lens.focal_y)
	cam_x, cam_y, cam_z = str(lens.cam_origin[0]), str(lens.cam_origin[1]), str(lens.cam_origin[2])
	cursor.execute('''INSERT INTO Scenic_Cameras (cameraId, worldId, ratio, origin, focalpoints, fov, skev_factor) '''+ \
			'''VALUES (\'%s\', \'%s\', %f, \'POINT Z (%s %s %s)\', \'POINT(%s %s)\', %s, %f);''' \
			%(camera_node.cam_id, world_name, camera_node.ratio, cam_x, cam_y, cam_z, focal_x, focal_y, lens.fov, lens.alpha))
	print("New camera inserted successfully.........")
	conn.commit()

def scenic_recognize(video_file, scenic_data_dir):
	### TODO: Read all attributes from the 
	### return formatted_result: {object: {bboxes: [[top_left:3d np.array, bottom_right:3d np.array]], 
 # 						  attributes: {*attr_name: *attr_value}}
 # 				}
	return {}	

def add_scenic_recognized_objs(conn, formatted_result, start_time, default_depth=True):
	clean_tables(conn)
	### TODO: Modify the following code to store the recognized formatted result into the database
	### TODO: Update the database schema to include all new fields
	for item_id in formatted_result:
		object_type = formatted_result[item_id]["object_type"]
		recognized_bboxes = np.array(formatted_result[item_id]["bboxes"])
		tracked_cnt = formatted_result[item_id]["tracked_cnt"]
		top_left = np.vstack((recognized_bboxes[:,0,0], recognized_bboxes[:,0,1]))
		if default_depth:
			top_left_depths = np.ones(len(recognized_bboxes))
		else:
			top_left_depths = self.__get_depths_of_points(recognized_bboxes[:,0,0], recognized_bboxes[:,0,1])
		
		# Convert bottom right coordinates to world coordinates
		bottom_right = np.vstack((recognized_bboxes[:,1,0], recognized_bboxes[:,1,1]))
		if default_depth:
			bottom_right_depths = np.ones(len(tracked_cnt))
		else:
			bottom_right_depths = self.__get_depths_of_points(recognized_bboxes[:,1,0], recognized_bboxes[:,1,1])
		
		top_left = np.array(top_left.T)
		bottom_right = np.array(bottom_right.T)
		obj_traj = []
		for i in range(len(top_left)):
			current_tl = top_left[i]
			current_br = bottom_right[i]
			obj_traj.append([current_tl.tolist(), current_br.tolist()])      
		
		scenic_bboxes_to_postgres(conn, item_id, object_type, "default_color" if item_id not in properties['color'] else properties['color'][item_id], start_time, tracked_cnt, obj_traj, type="yolov4")
		# bbox_to_tasm()

# Insert bboxes to postgres
def scenic_bboxes_to_postgres(conn, item_id, object_type, color, start_time, timestamps, bboxes, type='yolov3'):
	### TODO: Modify the following codes to add recognized scenic objects to the database
	if type == 'yolov3':
		timestamps = range(timestamps)

	converted_bboxes = [bbox_to_data3d(bbox) for bbox in bboxes]
	pairs = []
	deltas = []
	for meta_box in converted_bboxes:
		pairs.append(meta_box[0])
		deltas.append(meta_box[1:])
	postgres_timestamps = convert_timestamps(start_time, timestamps)
	create_or_insert_scenic_general_trajectory(conn, item_id, object_type, color, postgres_timestamps, bboxes, pairs)
	print(f"{item_id} saved successfully")


# Create general trajectory table
def create_or_insert_scenic_general_trajectory(conn, item_id, object_type, color, postgres_timestamps, bboxes, pairs):
	cursor = conn.cursor()
	'''
	Create and Populate A Trajectory table using mobilityDB.
	Now the timestamp matches, the starting time should be the meta data of the world
	Then the timestamp should be the timestamp regarding the world starting time
	'''
	
	### TODO: Modify the table fields to match scenic schema
	create_itemtraj_sql ='''CREATE TABLE IF NOT EXISTS Scenic_Item_General_Trajectory(
	itemId TEXT,
	objectType TEXT,
	color TEXT,
	trajCentroids tgeompoint,
	largestBbox stbox,
	PRIMARY KEY (itemId)
	);'''
	cursor.execute(create_itemtraj_sql)
	cursor.execute("CREATE INDEX IF NOT EXISTS traj_idx ON Scenic_Item_General_Trajectory USING GiST(trajCentroids);")
	conn.commit()
	#Creating table with the first item
	create_bboxes_sql ='''CREATE TABLE IF NOT EXISTS Scenic_General_Bbox(
	itemId TEXT,
	trajBbox stbox,
	FOREIGN KEY(itemId)
		REFERENCES Scenic_Item_General_Trajectory(itemId)
	);'''
	cursor.execute(create_bboxes_sql)
	cursor.execute("CREATE INDEX IF NOT EXISTS item_idx ON Scenic_General_Bbox(itemId);")
	cursor.execute("CREATE INDEX IF NOT EXISTS traj_bbox_idx ON Scenic_General_Bbox USING GiST(trajBbox);")
	conn.commit()
	#Insert the trajectory of the first item
	insert_scenic_general_trajectory(conn, item_id, object_type, color, postgres_timestamps, bboxes, pairs)


# Insert general trajectory
def insert_scenic_general_trajectory(conn, item_id, object_type, color, postgres_timestamps, bboxes, pairs):
    ### TODO: Modify the insert based on the new table schema
	#Creating a cursor object using the cursor() method
	cursor = conn.cursor()
	#Inserting bboxes into Bbox table
	insert_bbox_trajectory = ""
	insert_format = "INSERT INTO Scenic_General_Bbox (itemId, trajBbox) "+ \
	"VALUES (\'%s\',"  % (item_id)
	# Insert the item_trajectory separately
	insert_trajectory = "INSERT INTO Scenic_Item_General_Trajectory (itemId, objectType, color, trajCentroids, largestBbox) "+ \
	"VALUES (\'%s\', \'%s\', \'%s\', "  % (item_id, object_type, color)
	traj_centroids = "\'{"
	min_ltx, min_lty, min_ltz, max_brx, max_bry, max_brz = float('inf'), float('inf'), float('inf'), float('-inf'), float('-inf'), float('-inf')
	# max_ltx, max_lty, max_ltz, min_brx, min_bry, min_brz = float('-inf'), float('-inf'), float('-inf'), float('inf'), float('inf'), float('inf')
	for i in range(len(postgres_timestamps)):
		postgres_timestamp = postgres_timestamps[i]
		### Insert bbox
		# print(bboxes[i])
		tl, br = bboxes[i]
		min_ltx, min_lty, min_ltz, max_brx, max_bry, max_brz = min(tl[0], min_ltx), min(tl[1], min_lty), min(tl[2], min_ltz),\
			max(br[0], max_brx), max(br[1], max_bry), max(br[2], max_brz)
		# max_ltx, max_lty, max_ltz, min_brx, min_bry, min_brz = max(tl[0], max_ltx), max(tl[1], max_lty), max(tl[2], max_ltz),\
		#     min(br[0], min_brx), min(br[1], min_bry), min(br[2], min_brz)
		current_bbox_sql = "stbox \'STBOX ZT((%s, %s, %s, %s), (%s, %s, %s, %s))\');" \
		%(tl[0], tl[1], tl[2], postgres_timestamp, br[0], br[1], br[2], postgres_timestamp)
		insert_bbox_trajectory += insert_format + current_bbox_sql
		### Construct trajectory
		current_point = pairs[i]
		tg_pair_centroid = "POINT Z (%s %s %s)@%s," \
		%(str(current_point[0]), str(current_point[1]), str(current_point[2]), postgres_timestamp)
		traj_centroids += tg_pair_centroid
	traj_centroids = traj_centroids[:-1]
	traj_centroids += "}\', "
	insert_trajectory += traj_centroids
	insert_trajectory += "stbox \'STBOX Z((%s, %s, %s),"%(min_ltx, min_lty, min_ltz)\
		+"(%s, %s, %s))\'); "%(max_brx, max_bry, max_brz)
	cursor.execute(insert_trajectory)
	cursor.execute(insert_bbox_trajectory)
	# Commit your changes in the database
	conn.commit()
	