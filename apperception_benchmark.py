import unittest
from world import *
from world_executor import *
from video_util import *
from metadata_util import *
import json
import lens
import point
import cv2

import os
import shutil
#import tasm
import time

def establish_benchmark_world(name, units, video_file, metadata_id, lens_attrs, point_attrs, camera_attrs, recog_attrs):
	'''
	Inputs: lens_attrs = {'fov':..., 'cam_origin':..., 'skew_factor':...}
	'''
	new_world = World(name=name, units=units)
 
	# vs = cv2.VideoCapture(video_file)
	# frame = vs.read()
	# frame = frame[1]
	# cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)        # Create window with freedom of dimensions
	# cv2.resizeWindow('Frame', 384, 216)
	# initBB = cv2.selectROI("Frame", frame, fromCenter=False)
	# print(initBB)
	# cv2.destroyAllWindows()

	fov, res, cam_origin, skew_factor = lens_attrs['fov'], [3840, 2160], lens_attrs['cam_origin'], lens_attrs['skew_factor']
	cam_lens = lens.PinholeLens(res, cam_origin, fov, skew_factor)
	# print("world coordinate #1")
	# print(cam_lens.pixel_to_world((12, 1619), 1))
	# print("world coordinate #2")
	# print(cam_lens.pixel_to_world((12+3325, 1619+476), 1))
	pt_id, cam_id, x, y, z, time, pt_type = point_attrs['p_id'], point_attrs['cam_id'], point_attrs['x'], point_attrs['y'], point_attrs['z'], point_attrs['time'], point_attrs['type']
	pt = point.Point(pt_id, cam_id, x, y, z, time, pt_type)

	ratio = camera_attrs['ratio']
	algo, tracker_type = recog_attrs['algo'], recog_attrs['tracker']
	recog_world = new_world.camera(cam_id, pt, ratio, video_file, metadata_id, cam_lens).recognize(cam_id, algo, tracker_type)
	# recog_world.execute()
	return new_world

name = 'traffic_scene'
units = 'metrics'
video_file = './traffic-001.mp4'
lens_attrs = {
	'fov': 120, 
	'cam_origin': (0, 0, 0), 
	'skew_factor': 0}
point_attrs = {
	'p_id': 'p1', 
	'cam_id': 'cam1', 
	'x': 0,
	'y': 0, 
	'z': 0,
	'time': None, 
	'type':'pos'
}
camera_attrs = {
	'ratio': 0.5
}
recog_attrs = {
	'algo': 'Yolo', 
	'tracker': 'multi'
}


traffic_world = establish_benchmark_world(name, units, video_file, "traffic-scene", lens_attrs, point_attrs, camera_attrs, recog_attrs)


def main():
	### Get the intersection volume, either provided by user
	### or let the user select a 2d bbox
	start = time.perf_counter()
	volume = "stbox \'STBOX Z((0.01082532, 2.59647246, 0),(3.01034039, 3.35985782, 2))\'"		
	filtered_world = traffic_world.predicate(lambda obj:obj.object_type == "car").predicate(lambda obj:obj.location in volume, {"volume":volume})
	filtered_world = filtered_world.interval([30*300,30*1200])	
 
	### to get the video over the entire trajectory(amber case)
	filtered_ids = filtered_world.selectkey(distinct = True).execute()
	id_time = time.perf_counter()
	print("fetch id time is:", id_time-start)
	print("filtered_ids are", filtered_ids)
	print(len(filtered_ids))
	if filtered_ids:
		id_array = [filtered_id[0] for filtered_id in filtered_ids]
		print("filtered_ids are", len(id_array))
		trajectory = traffic_world.predicate(lambda obj: obj.object_id in id_array, {"id_array":id_array}).get_trajectory(distinct=True).execute()
		print(trajectory[0])
		traj_time = time.perf_counter()
		print("fetch traj time is:", traj_time-start)
		entire_video = traffic_world.predicate(lambda obj: obj.object_id in id_array, {"id_array":id_array}).get_video()
		print(entire_video.execute())

if __name__ == '__main__':
	main()
