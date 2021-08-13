import unittest
from video_context import *
from video_context_executor import *
from video_util import *
import json
import lens
import point
import camera as cam

test_executor = VideoContextExecutor()
test_context = VideoContext(name = 'traffic_scene', units = 'metric')
# Connect depending on docker or mobilitydb
test_context.connect_db(user="docker", password="docker", database_name="mobilitydb")

# Test simple queries using Context class
class TestVideoContextMethods(unittest.TestCase):
	def test_create_video_context(self):
		# Initialize a new context for executor
		test_executor.context(test_context)

		print(test_executor.execute())

	def test_create_world_with_cameras(self):
		test_context.clear()
		# Initialize a new context for executor
		test_executor.context(test_context)
		
		# Add camera
		# Create pinhole lens
		video = cam.VideoPhysical('cars.mp4')
		cam_lens = lens.PinholeLens(120, video.resolution, (0, 0, 0), 0)

		# Let's say that the camera is in the origin of this "new world"
		pt = point.Point('p1', 'cam1', 0, 0, 0, None, 'pos')
		camera_1 = test_context.camera('cam1', pt, 0.5, video, cam_lens)

		print(test_executor.execute())

	def test_object_rec(self):
		test_context.clear()
		# Initialize a new context for executor
		test_executor.context(test_context)
		
		# Add camera
		# Create pinhole lens
		video = cam.VideoPhysical('cars.mp4')
		cam_lens = lens.PinholeLens(120, video.resolution, (0, 0, 0), 0)

		# Let's say that the camera is in the origin of this "new world"
		pt = point.Point('p1', 'cam1', 0, 0, 0, None, 'pos')
		camera_1 = test_context.camera('cam1', pt, 0.5, video, cam_lens)

		# Perform object recognition and add objects
		camera_1.recognize('Yolo', 'multi')

		print(test_executor.execute())

if __name__ == '__main__':
	unittest.main()