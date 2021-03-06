import ast
import psycopg2
import numpy as np
import datetime 
import cv2
from object_tracker import yolov4_deepsort_video_track


def video_data_to_tasm(video_file, metadata_id, t):
	t.store(video_file, metadata_id)

def metadata_to_tasm(formatted_result, metadata_id, t):
	import tasm
	metadata_info = []
	bound_width = lambda x : min(max(0, x), 3840)
	bound_height = lambda y: min(max(0, y), 2160)
	for obj, info in formatted_result.items():
		object_type = info['object_type']
		for bbox, frame in zip(info['bboxes'], info['tracked_cnt']):
			x1 = bound_width(bbox[0][0])
			y1 = bound_height(bbox[0][1])
			x2 = bound_width(bbox[1][0])
			y2 = bound_height(bbox[1][1])
			if frame < 0 or x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0:
				import pdb; pdb.set_trace()
			metadata_info.append(tasm.MetadataInfo(metadata_id, object_type, frame, x1, y1, x2, y2))
			metadata_info.append(tasm.MetadataInfo(metadata_id, obj, frame, x1, y1, x2, y2))

	t.add_bulk_metadata(metadata_info)

def convert_datetime_to_frame_num(start_time, date_times):
	
	return [(t.replace(tzinfo = None) - start_time).total_seconds() for t in date_times] 

def get_video_roi(file_name, cam_video_file, rois, times):
	"""
	Get the region of interest from the video, based on bounding box points in
	video coordinates.
	
	Args:
		file_name: String of file name to save video as
		rois: A list of bounding boxes
		time_intervals: A list of time intervals of which frames
	"""

	rois = np.array(rois).T
	print(rois.shape)
	len_x, len_y = np.max(rois.T[2] - rois.T[0]), np.max(rois.T[3] - rois.T[1])
	# len_x, len_y  = np.max(rois.T[0][1] - rois.T[0][0]), np.max(rois.T[1][1] - rois.T[1][0])

	len_x = int(round(len_x))
	len_y = int(round(len_y))
	# print(len_x)
	# print(len_y)
	vid_writer = cv2.VideoWriter(file_name, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 30, (len_x, len_y))
	# print("rois")
	# print(rois)
	start_time = int(times[0])
	cap = cv2.VideoCapture(cam_video_file)
	frame_cnt = 0
	while(cap.isOpened()):
		# Capture frame-by-frame
		ret, frame = cap.read()
		if frame_cnt in times and ret:
			i = frame_cnt - start_time
			if i >= len(rois):
				print("incorrect length:", len(rois))
				break
			current_roi = rois[i]

			b_x, b_y, e_x, e_y = current_roi
			b_x, b_y = max(0, b_x), max(0, b_y)
			# e_x, e_y = current_roi[1]
			e_x, e_y = max(0, e_x), max(0, e_y)
			diff_y, diff_x =int(abs(e_y - b_y)), int(abs(e_x - b_x))
			pad_y = int((len_y - diff_y) // 2)
			pad_x = int((len_x - diff_x) // 2)

			# print("padding")
			# print(pad_y)
			# print(pad_x)
			roi_byte = frame[int(b_y):int(e_y), int(b_x): int(e_x), :]
			

			roi_byte = np.pad(roi_byte, pad_width = [(pad_y, len_y - diff_y - pad_y), (pad_x, len_x - diff_x - pad_x), (0, 0)])
			frame = cv2.cvtColor(roi_byte, cv2.COLOR_RGB2BGR)

			
			vid_writer.write(roi_byte)
		frame_cnt += 1
		if not ret:
			break

	vid_writer.release()

def create_or_insert_world_table(conn, name, units):
	#Creating a cursor object using the cursor() method
	cursor = conn.cursor()
	'''
	Create and Populate A world table with the given world object.
	'''
	#Doping Worlds table if already exists. TODO: For testing purpose only
	cursor.execute("DROP TABLE IF EXISTS Worlds;")
	#Creating table with the first world
	sql = '''CREATE TABLE IF NOT EXISTS Worlds(
	worldId TEXT PRIMARY KEY,
	units TEXT
	);'''
	cursor.execute(sql)
	print("Worlds Table created successfully........")
	insert_world(conn, name, units)
	return sql

# Helper function to insert the world
def insert_world(conn, name, units):
	#Creating a cursor object using the cursor() method
	cursor = conn.cursor()
	cursor.execute('''INSERT INTO Worlds (worldId, units) '''+ \
			'''VALUES (\'%s\',  \'%s\');''' \
			%(name, units))
	print("New world inserted successfully........")
	#Insert the existing cameras of the current world into the camera table
	conn.commit()

# Create a camera table
def create_or_insert_camera_table(conn, world_name, camera):
	#Creating a cursor object using the cursor() method
	cursor = conn.cursor()
	'''
	Create and Populate A camera table with the given camera object.
	'''
	#Doping Cameras table if already exists. TODO: For testing purpose only
	cursor.execute("DROP TABLE IF EXISTS Cameras")
	#Creating table with the first camera
	sql = '''CREATE TABLE IF NOT EXISTS Cameras(
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
	insert_camera(conn, world_name, camera)
	return sql

# Helper function to insert the camera
def insert_camera(conn, world_name, camera_node):
	#Creating a cursor object using the cursor() method
	cursor = conn.cursor()
	lens = camera_node.lens
	focal_x = str(lens.focal_x)
	focal_y = str(lens.focal_y)
	cam_x, cam_y, cam_z = str(lens.cam_origin[0]), str(lens.cam_origin[1]), str(lens.cam_origin[2])
	cursor.execute('''INSERT INTO Cameras (cameraId, worldId, ratio, origin, focalpoints, fov, skev_factor) '''+ \
			'''VALUES (\'%s\', \'%s\', %f, \'POINT Z (%s %s %s)\', \'POINT(%s %s)\', %s, %f);''' \
			%(camera_node.cam_id, world_name, camera_node.ratio, cam_x, cam_y, cam_z, focal_x, focal_y, lens.fov, lens.alpha))
	print("New camera inserted successfully.........")
	conn.commit()

# Default object recognition (YOLOv3)
def recognize(video_file, recog_algo = "", tracker_type = "default", customized_tracker = None):
	# recognition = item.ItemRecognition(recog_algo = recog_algo, tracker_type = tracker_type, customized_tracker = customized_tracker)
	# return recognition.video_item_recognize(video.byte_array)
	return yolov4_deepsort_video_track(video_file)	


def add_recognized_objs(conn, lens, formatted_result, start_time, properties={'color':{}}, default_depth=True):
	clean_tables(conn)
	for item_id in formatted_result:
		object_type = formatted_result[item_id]["object_type"]
		recognized_bboxes = np.array(formatted_result[item_id]["bboxes"])
		tracked_cnt = formatted_result[item_id]["tracked_cnt"]
		top_left = np.vstack((recognized_bboxes[:,0,0], recognized_bboxes[:,0,1]))
		if default_depth:
			top_left_depths = np.ones(len(recognized_bboxes))
		else:
			top_left_depths = self.__get_depths_of_points(recognized_bboxes[:,0,0], recognized_bboxes[:,0,1])
		top_left = lens.pixels_to_world(top_left, top_left_depths)
		
		# Convert bottom right coordinates to world coordinates
		bottom_right = np.vstack((recognized_bboxes[:,1,0], recognized_bboxes[:,1,1]))
		if default_depth:
			bottom_right_depths = np.ones(len(tracked_cnt))
		else:
			bottom_right_depths = self.__get_depths_of_points(recognized_bboxes[:,1,0], recognized_bboxes[:,1,1])
		bottom_right = lens.pixels_to_world(bottom_right, bottom_right_depths)
		
		top_left = np.array(top_left.T)
		bottom_right = np.array(bottom_right.T)
		obj_traj = []
		for i in range(len(top_left)):
			current_tl = top_left[i]
			current_br = bottom_right[i]
			obj_traj.append([current_tl.tolist(), current_br.tolist()])      
		
		bbox_to_postgres(conn, item_id, object_type, "default_color" if item_id not in properties['color'] else properties['color'][item_id], start_time, tracked_cnt, obj_traj, type="yolov4")
		# bbox_to_tasm()
	
# Helper function to convert the timestam to the timestamp formula pg-trajectory uses
def convert_timestamps(start_time, timestamps):
	return [str(start_time + datetime.timedelta(seconds=t)) for t in timestamps]

# Helper function to convert trajectory to centroids
def bbox_to_data3d(bbox):
	'''
	Compute the center, x, y, z delta of the bbox
	'''
	tl, br = bbox
	x_delta = (br[0] - tl[0])/2
	y_delta = (br[1] - tl[1])/2
	z_delta = (br[2] - tl[2])/2
	center = (tl[0] + x_delta, tl[1] + y_delta, tl[2] + z_delta)
	
	return center, x_delta, y_delta, z_delta

# Insert bboxes to postgres
def bbox_to_postgres(conn, item_id, object_type, color, start_time, timestamps, bboxes, type='yolov3'):
	if type == 'yolov3':
		timestamps = range(timestamps)

	converted_bboxes = [bbox_to_data3d(bbox) for bbox in bboxes]
	pairs = []
	deltas = []
	for meta_box in converted_bboxes:
		pairs.append(meta_box[0])
		deltas.append(meta_box[1:])
	postgres_timestamps = convert_timestamps(start_time, timestamps)
	create_or_insert_general_trajectory(conn, item_id, object_type, color, postgres_timestamps, bboxes, pairs)
	print(f"{item_id} saved successfully")

def clean_tables(conn):
	cursor = conn.cursor()
	cursor.execute("DROP TABLE IF EXISTS General_Bbox;")
	cursor.execute("DROP TABLE IF EXISTS Item_General_Trajectory;")
	conn.commit()

# Create general trajectory table
def create_or_insert_general_trajectory(conn, item_id, object_type, color, postgres_timestamps, bboxes, pairs):
	#Creating a cursor object using the cursor() method
	cursor = conn.cursor()
	'''
	Create and Populate A Trajectory table using mobilityDB.
	Now the timestamp matches, the starting time should be the meta data of the world
	Then the timestamp should be the timestamp regarding the world starting time
	'''
	
	#Creating table with the first item
	create_itemtraj_sql ='''CREATE TABLE IF NOT EXISTS Item_General_Trajectory(
	itemId TEXT,
	objectType TEXT,
	color TEXT,
	trajCentroids tgeompoint,
	largestBbox stbox,
	PRIMARY KEY (itemId)
	);'''
	cursor.execute(create_itemtraj_sql)
	cursor.execute("CREATE INDEX IF NOT EXISTS traj_idx ON Item_General_Trajectory USING GiST(trajCentroids);")
	conn.commit()
	#Creating table with the first item
	create_bboxes_sql ='''CREATE TABLE IF NOT EXISTS General_Bbox(
	itemId TEXT,
	trajBbox stbox,
	FOREIGN KEY(itemId)
		REFERENCES Item_General_Trajectory(itemId)
	);'''
	cursor.execute(create_bboxes_sql)
	cursor.execute("CREATE INDEX IF NOT EXISTS item_idx ON General_Bbox(itemId);")
	cursor.execute("CREATE INDEX IF NOT EXISTS traj_bbox_idx ON General_Bbox USING GiST(trajBbox);")
	conn.commit()
	#Insert the trajectory of the first item
	insert_general_trajectory(conn, item_id, object_type, color, postgres_timestamps, bboxes, pairs)


# Insert general trajectory
def insert_general_trajectory(conn, item_id, object_type, color, postgres_timestamps, bboxes, pairs):
	#Creating a cursor object using the cursor() method
	cursor = conn.cursor()
	#Inserting bboxes into Bbox table
	insert_bbox_trajectory = ""
	insert_format = "INSERT INTO General_Bbox (itemId, trajBbox) "+ \
	"VALUES (\'%s\',"  % (item_id)
	# Insert the item_trajectory separately
	insert_trajectory = "INSERT INTO Item_General_Trajectory (itemId, objectType, color, trajCentroids, largestBbox) "+ \
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
	
def merge_trajectory(item_id, new_postgres_timestamps, new_bboxes, new_pairs):
	### Fetch the timestamps of the current trajectory from the database
	### Filter out the already had timestamp from the new timestamps
	### Construct the adding trajectory
	### Calling the merge function of mobilitydb
	### do the same thing for the bboxes
	return
	
def fetch_camera(conn, world_id, cam_id = []):
	cursor = conn.cursor()
	
	if cam_id == []:
		query = '''SELECT cameraId, ratio, ST_X(origin), ST_Y(origin), ST_Z(origin), ST_X(focalpoints), ST_Y(focalpoints), fov, skev_factor ''' \
		 + '''FROM Cameras WHERE worldId = \'%s\';''' %world_id
	else:
		query = '''SELECT cameraId, ratio, ST_X(origin), ST_Y(origin), ST_Z(origin), ST_X(focalpoints), ST_Y(focalpoints), fov, skev_factor ''' \
		 + '''FROM Cameras WHERE cameraId IN (\'%s\') AND worldId = \'%s\';''' %(','.join(cam_id), world_id)
	cursor.execute(query)
	return cursor.fetchall()