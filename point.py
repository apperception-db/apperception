class Point:

	def __init__(self, point_id: str, object_id: str, x: float, 
		y: float, z: float, time: float, point_type: str):
		''' 
		Initializes an Point given coordinates, time, type and associated point ID
		and object ID. 
		''' 
		self.point_id = point_id
		self.object_id = object_id
		self.coordinate = (x, y, z)
		self.time = time
		self.point_type = point_type


