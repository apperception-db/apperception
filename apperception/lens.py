import numpy as np
from math import radians

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
		cam_x, cam_y, cam_z = cam_origin

		yaw, pitch, roll = np.deg2rad(yaw), np.deg2rad(pitch), np.deg2rad(roll)
		
		self.fov = field_of_view
		self.focal_width = width/2*np.tan(field_of_view*np.pi/360)
		self.focal_height = height/2*np.tan(field_of_view*np.pi/360)
		print("focal_width", self.focal_width)
		print("focal_height", self.focal_height)
		self.alpha = skew_factor
		self.scaling_transform = np.linalg.inv(np.asarray(
								[[self.focal_width, self.alpha, width/2], 
								 [0, self.focal_height, height/2],
								 [0, 0, 1],
								]))
		print("scaling matrix is", self.scaling_transform)
		
		R_1 = np.cos(pitch)*np.cos(yaw)
		R_2 = np.sin(roll)*np.sin(pitch)*np.cos(yaw) + np.cos(roll)*np.sin(yaw)
		R_3 = np.sin(roll)*np.sin(yaw) - np.cos(roll)*np.sin(pitch)*np.cos(yaw)
		R_4 = -np.cos(pitch)*np.sin(yaw)
		R_5 = np.cos(roll)*np.cos(yaw)-np.sin(roll)*np.sin(pitch)*np.sin(yaw)
		R_6 = np.cos(roll)*np.sin(pitch)*np.sin(yaw) + np.sin(roll)*np.cos(yaw)
  
		R_7 = np.sin(pitch)
		R_8 = np.sin(roll)*-np.cos(pitch)
		R_9 = np.cos(roll)*np.cos(pitch)

		self.rotational_transform = np.asarray([[R_1, R_2, R_3, cam_x],
			[R_4, R_5, R_6, cam_y], 
			[R_7, R_8, R_9, cam_z],
			[0, 0, 0, 1]
			])

		self.inv_scaling_transform = np.linalg.inv(self.scaling_transform)
		self.inv_rotational_transform = np.linalg.inv(self.rotational_transform)
  
	def pixel_to_world(self, pixel_coord, depth):
		"""
		Translate pixel coordinates to world coordinates. 
		"""       
		x, y = pixel_coord
		pixel = np.array([[x], [y], [1]])
		scaled_matrix = self.scaling_transform @ pixel
		scaled_pixels = np.asarray([scaled_matrix[0], scaled_matrix[1], [depth], [1]])
		return self.rotational_transform @ scaled_pixels

	def pixels_to_world(self, pixel_coords, depths):
		"""
		Translate multiple pixel coordinates to world coordinates. 
		"""
		x, y =  pixel_coords
		pixels = np.asarray([x, y, np.ones(len(depths))])
		scaled_matrix = self.scaling_transform @ pixels
		scaled_pixels = np.asarray([scaled_matrix[0], scaled_matrix[1], depths, np.ones(len(depths))])
		return self.rotational_transform @ scaled_pixels 

	def world_to_pixel(self, world_coord):
		"""
		Translate world coordinates to pixel coordinates
		"""
		x, y, z, w = world_coord
		world_pixel = np.asarray([[x], [y], [z], [w]])
		scaled_pixels = self.inv_rotational_transform @ world_pixel
		print("scaled_pixels after inv rotation", scaled_pixels)
		original_pixel = self.inv_scaling_transform @ np.asarray([scaled_pixels[0], scaled_pixels[1], [1]])
		return original_pixel

	def world_to_pixels(self, world_coords):
		"""
		Translate world coordinates to pixel coordinates
		"""
		scaled_pixels = self.inv_rotational_transform @ world_coords
		original_pixels = self.inv_scaling_transform @ np.asarray([scaled_pixels[0], scaled_pixels[1], np.ones(len(scaled_pixels[0]))])
		return original_pixels    


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
		self.focal_x = (x/2)/np.tan(radians(field_of_view/2)) 
		self.focal_y = (y/2)/np.tan(radians(field_of_view/2)) 
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