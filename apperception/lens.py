import numpy as np
import math
import cv2

class Lens:
	def __init__(self, resolution, cam_origin):
		"""
		Construct a lens for the camera that translates to 3D world coordinates.

		Args:
			field_of_view: Angle of field of view of camera
			resolution: Tuple of video resolution
			cam_origin: Points of where camera is located in the world
			skew_factor: (Optional) Float factor to correct shearness of camera  
		"""
		x, y = resolution
		self.cam_origin = cam_origin
		cam_x, cam_y = cam_origin
	
	def pixel_to_world(self, pixel_coord, depth):
		"""
		Translate pixel coordinates to world coordinates. 
		""" 
		return None
	
	def world_to_pixel(self, world_coord, depth):
		"""
		Translate world coordinates to pixel coordinates
		"""
		return None

class VRLens(Lens):
	def __init__(self, resolution, cam_origin, yaw, roll, pitch, field_of_view, skew_factor=0):
		"""
		Construct a lens for the camera that translates to 3D world, spherical 
		coordinates.

		Args:
			field_of_view: Angle of field of view of camera
			resolution: Tuple of video resolution
			cam_origin: Points of where camera is located in the world
			skew_factor: (Optional) Float factor to correct shearness of camera   
		"""
		width, height = resolution
		self.cam_origin = cam_origin
		self.fov = field_of_view
		self.alpha = skew_factor
		rotation = yaw, roll, pitch
		self.intrinsic_matrix = self.create_intrinsic_matrix(width, height, self.fov)
		self.extrinsic_matrix = self.create_extrinsic_matrix(cam_origin, rotation)

	def create_matrix(self, location, rotation):
		"""Creates a transformation matrix to convert points in the 3D world
		coordinate space with respect to the object.
		Use the transform_points function to transpose a given set of points
		with respect to the object.
		Args:
			location (:py:class:`.Location`): The location of the object
				represented by the transform.
			rotation (:py:class:`.Rotation`): The rotation of the object
				represented by the transform.
		Returns:
			A 4x4 numpy matrix which represents the transformation matrix.
		"""
		matrix = np.identity(4)
		x, y, z = location
		yaw, roll, pitch = rotation
		cy = math.cos(np.radians(yaw))
		sy = math.sin(np.radians(yaw))
		cr = math.cos(np.radians(roll))
		sr = math.sin(np.radians(roll))
		cp = math.cos(np.radians(pitch))
		sp = math.sin(np.radians(pitch))
		matrix[0, 3] = x
		matrix[1, 3] = y
		matrix[2, 3] = z
		matrix[0, 0] = (cp * cy)
		matrix[0, 1] = (cy * sp * sr - sy * cr)
		matrix[0, 2] = -1 * (cy * sp * cr + sy * sr)
		matrix[1, 0] = (sy * cp)
		matrix[1, 1] = (sy * sp * sr + cy * cr)
		matrix[1, 2] = (cy * sr - sy * sp * cr)
		matrix[2, 0] = (sp)
		matrix[2, 1] = -1 * (cp * sr)
		matrix[2, 2] = (cp * cr)
		return matrix
  
	def create_intrinsic_matrix(self, width, height, fov):
		k = np.identity(3)
		# We use width - 1 and height - 1 to find the center column and row
		# of the image, because the images are indexed from 0.

		# Center column of the image.
		k[0, 2] = (width - 1) / 2.0
		# Center row of the image.
		k[1, 2] = (height - 1) / 2.0
		# Focal length.
		k[0, 0] = k[1, 1] = (width - 1) / (2.0 * np.tan(fov * np.pi / 360.0))
		return k

	def create_extrinsic_matrix(self, location, rotation):
		transform = self.create_matrix(location, rotation)
		to_unreal_transform = np.array(
				[[0, 0, 1, 0], [1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
		return transform @ to_unreal_transform

	def pixel_to_world(self, pixel_coord, depth):
		"""
		Translate pixel coordinates to world coordinates. 
		"""
		transformed_3D_pos = np.dot(np.linalg.inv(self.intrinsic_matrix),pixel_coord)
		transformed_3D_pos = np.asarray([[transformed_3D_pos[0]], [transformed_3D_pos[1]], [transformed_3D_pos[2]], [1]])
		position_vector = self.extrinsic_matrix @ transformed_3D_pos
		return position_vector

	def pixels_to_world(self, pixel_coords, depths):
		"""
		Translate multiple pixel coordinates to world coordinates. 
		"""
		x, y =  pixel_coords
		pixels = np.asarray([x, y, depths])
		transformed_3D_pos = np.dot(np.linalg.inv(self.intrinsic_matrix),pixels)
		transformed_3D_pos = np.asarray([transformed_3D_pos[0], transformed_3D_pos[1], depths, np.ones(len(depths))])
		position_vector = self.extrinsic_matrix @ transformed_3D_pos
		return position_vector

	def world_to_pixel(self, world_coord):
		"""
		Translate world coordinates to pixel coordinates
		"""
		x, y, z = world_coord
		world_pixel = np.asarray([[x], [y], [z], [1]])
		transformed_3D_pos = np.dot(np.linalg.inv(self.extrinsic_matrix),
								world_pixel)
		position_2D = np.dot(self.intrinsic_matrix, transformed_3D_pos[:3])
		return position_2D

	def world_to_pixels(self, world_coords):
		"""
		Translate world coordinates to pixel coordinates
		"""
		x, y, z = world_coords
		world_pixel = np.asarray([[x], [y], [z], np.ones(len(x))])
		transformed_3D_pos = np.dot(np.linalg.inv(self.extrinsic_matrix),
								world_pixel)
		position_2D = np.dot(self.intrinsic_matrix, transformed_3D_pos[:3])
		return position_2D    


class PinholeLens(Lens):
	# TODO: (@Vanessa) change all the places where pinhole lens appears and change arguments
	def __init__(self, resolution, cam_origin, field_of_view, skew_factor):
		"""
		Construct a lens for the camera that translates to 3D world coordinates.
		
		Args:
			field_of_view: Angle of field of view of camera
			resolution: Tuple of video resolution
			cam_origin: Points of where camera is located in the world
			skew_factor: (Optional) Float factor to correct shearness of camera  
			depth: Float of depth of view from the camera
		"""
		self.fov = field_of_view
		x, y = resolution
		self.focal_x = (x/2)/np.tan(math.radians(field_of_view/2)) 
		self.focal_y = (y/2)/np.tan(math.radians(field_of_view/2)) 
		self.cam_origin = cam_origin
		cam_x, cam_y, cam_z = cam_origin
		self.alpha = skew_factor
		self.inv_transform = np.linalg.inv(np.matrix([[self.focal_x, self.alpha, cam_x], 
									[0, self.focal_y, cam_y],
									[0, 0, 1]
								   ]))
		self.transform = np.matrix([[self.focal_x, self.alpha, cam_x, 0], 
									[0, self.focal_y, cam_y, 0],
									[0, 0, 1, 0]
								   ])
		
	def pixel_to_world(self, pixel_coord, depth):
		"""
		Translate pixel coordinates to world coordinates. 
		"""
		x, y = pixel_coord
		pixel = np.matrix([[x], [y], [depth]])
		return (self.inv_transform @ pixel).flatten().tolist()[0]

	def pixels_to_world(self, pixel_coords, depths):
		"""
		Translate multiple pixel coordinates to world coordinates. 
		"""
		x, y =  pixel_coords
		pixels = np.matrix([x, y, depths])
		return self.inv_transform @ pixels 
	
	def world_to_pixel(self, world_coord):
		"""
		Translate world coordinates to pixel coordinates
		"""
		x, y, z = world_coord
		world_pixel = np.matrix([[x], [y], [z], [1]])
		return self.transform @ world_pixel

	def world_to_pixels(self, world_coords):
		"""
		Translate world coordinates to pixel coordinates
		"""
		x, y, z = world_coords
		world_pixel = np.matrix([x, y, z, np.ones(len(x))])
		return self.transform @ world_pixel

