from __future__ import annotations

from math import radians

import numpy as np


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

    def pixels_to_world(self, pixel_coords, depths):
        """
        Translate multiple pixel coordinates to world coordinates.
        """
        return None

    def world_to_pixel(self, world_coord, depth):
        """
        Translate world coordinates to pixel coordinates
        """
        return None

    def __eq__(self, other):
        return self.__dict__ == other.__dict__


class VRLens(Lens):
    def __init__(self, resolution, cam_origin, yaw, roll, pitch):
        """
        Construct a lens for the camera that translates to 3D world, spherical
        coordinates.

        Args:
                field_of_view: Angle of field of view of camera
                resolution: Tuple of video resolution
                cam_origin: Points of where camera is located in the world
                skew_factor: (Optional) Float factor to correct shearness of camera
        """
        x, y = resolution
        self.cam_origin = cam_origin
        cam_x, cam_y, cam_z = cam_origin

        yaw, pitch, roll = np.deg2rad(yaw), np.deg2rad(pitch), np.deg2rad(roll)
        # Transformation 1
        # X_1, X_2, X_3 = np.cos(pitch)*np.cos(yaw), np.cos(pitch)*np.sin(yaw), -np.sin(pitch)

        # Y_1 = np.cos(yaw)*np.sin(pitch)*np.sin(roll) - np.sin(yaw)*np.cos(roll)
        # Y_2 = np.sin(yaw)*np.sin(pitch)*np.sin(roll) + np.cos(yaw)*np.cos(roll)
        # Y_3 = np.cos(pitch)*np.sin(roll)

        # Z_1 = np.cos(yaw)*np.sin(pitch)*np.cos(roll) + np.sin(yaw)*np.sin(roll)
        # Z_2 = np.sin(yaw)*np.sin(pitch)*np.cos(roll) - np.cos(yaw)*np.sin(roll)
        # Z_3 = np.cos(pitch)*np.cos(roll)

        # self.transform = np.matrix([[X_1, Y_1, Z_1, cam_x],
        # 	[X_2, Y_2, Z_2, cam_y],
        # 	[X_3, Y_3, Z_3, cam_z],
        # 	[0, 0, 0, 1]
        # 	])

        # Transformation 2
        # z = yaw, y = pitch, x = roll
        # R_1, R_2, R_3 = np.cos(pitch)*np.cos(yaw), np.cos(pitch)*np.sin(yaw), np.sin(pitch)
        # R_4 = np.sin(roll)*np.sin(pitch)*np.cos(yaw) - np.cos(roll)*np.sin(yaw)
        # R_5 = np.sin(roll)*np.sin(pitch)*np.sin(yaw) + np.cos(roll)*np.cos(yaw)
        # R_6 = np.sin(roll)*np.cos(pitch)
        # R_7 = np.cos(roll)*np.sin(pitch)*np.cos(yaw) - np.sin(roll)*np.sin(yaw)
        # R_8 = np.sin(roll)*np.cos(yaw) + np.cos(roll)*np.sin(pitch)*np.sin(yaw)
        # R_9 = np.cos(roll)*np.cos(pitch)

        # self.transform = np.matrix([[R_1, R_2, R_3, cam_x],
        # 	[R_4, R_5, R_6, cam_y],
        # 	[R_7, R_8, R_9, cam_z],
        # 	[0, 0, 0, 1]
        # 	])

        # Transformation 3
        # z = yaw, y = pitch, x = roll
        # R_1, R_2, R_3 = np.cos(pitch)*np.cos(yaw), np.cos(pitch)*np.sin(yaw), np.sin(pitch)
        # R_4 = np.sin(roll)*np.sin(pitch)*np.cos(yaw) - np.cos(roll)*np.sin(yaw)
        # R_5 = np.sin(roll)*np.sin(pitch)*np.sin(yaw) + np.cos(roll)*np.cos(yaw)
        # R_6 = np.sin(roll)*-np.cos(pitch)
        # R_7 = -np.cos(roll)*np.sin(pitch)*np.cos(yaw) - np.sin(roll)*np.sin(yaw)
        # R_8 = np.sin(roll)*np.cos(yaw) - np.cos(roll)*np.sin(pitch)*np.sin(yaw)
        # R_9 = np.cos(roll)*np.cos(pitch)

        # rotation_mat = np.matrix([[R_1, R_2, R_3],
        # 	[R_4, R_5, R_6],
        # 	[R_7, R_8, R_9]])

        # cam_org_vec = np.matrix([[cam_x], [cam_y], [cam_z]])
        # self.col_vec = np.ravel(rotation_mat @ cam_org_vec)
        # col_x, col_y, col_z = self.col_vec
        # self.transform = np.matrix([[R_1, R_2, R_3, -col_x],
        # 	[R_4, R_5, R_6, -col_y],
        # 	[R_7, R_8, R_9, -col_z],
        # 	[0, 0, 0, 1]
        # 	])

        # Transformation 4
        # X_1, X_2, X_3 = np.cos(pitch)*np.cos(yaw), np.cos(pitch)*np.sin(yaw), -np.sin(pitch)

        # Y_1 = np.cos(yaw)*np.sin(pitch)*np.sin(roll) - np.sin(yaw)*np.cos(roll)
        # Y_2 = np.sin(yaw)*np.sin(pitch)*np.sin(roll) + np.cos(yaw)*np.cos(roll)
        # Y_3 = np.cos(pitch)*np.sin(roll)

        # Z_1 = np.cos(yaw)*np.sin(pitch)*np.cos(roll) + np.sin(yaw)*np.sin(roll)
        # Z_2 = np.sin(yaw)*np.sin(pitch)*np.cos(roll) - np.cos(yaw)*np.sin(roll)
        # Z_3 = np.cos(pitch)*np.cos(roll)

        # rotation_mat = np.matrix([[X_1, Y_1, Z_1],
        # 	[X_2, Y_2, Z_2],
        # 	[X_3, Y_3, Z_3]])
        # cam_org_vec = np.matrix([[cam_x], [cam_y], [cam_z]])
        # self.col_vec = np.ravel(rotation_mat @ cam_org_vec)
        # col_x, col_y, col_z = self.col_vec
        # self.transform = np.matrix([[X_1, Y_1, Z_1, col_x],
        # 	[X_2, Y_2, Z_2, col_y],
        # 	[X_3, Y_3, Z_3, col_z],
        # 	[0, 0, 0, 1]
        # 	])

        # Transformation 5 -- Lefthanded rotation matrix
        R_1, R_2, R_3 = np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), -np.sin(pitch)
        R_4 = np.sin(roll) * np.sin(pitch) * np.cos(yaw) - np.cos(roll) * np.sin(yaw)
        R_5 = np.sin(roll) * np.sin(pitch) * np.sin(yaw) + np.cos(roll) * np.cos(yaw)
        R_6 = np.sin(roll) * np.cos(pitch)

        R_7 = np.cos(roll) * np.sin(pitch) * np.cos(yaw) + np.sin(roll) * np.sin(yaw)
        R_8 = np.cos(roll) * np.sin(pitch) * np.sin(yaw) - np.sin(roll) * np.cos(yaw)
        R_9 = np.cos(roll) * np.cos(pitch)

        rotation_mat = np.matrix([[R_1, R_2, R_3], [R_4, R_5, R_6], [R_7, R_8, R_9]])
        cam_org_vec = np.matrix([[cam_x], [cam_y], [cam_z]])
        self.col_vec = np.ravel(rotation_mat @ cam_org_vec)
        col_x, col_y, col_z = self.col_vec
        self.transform = np.matrix(
            [
                [R_1, R_2, R_3, -col_x],
                [R_4, R_5, R_6, -col_y],
                [R_7, R_8, R_9, -col_z],
                [0, 0, 0, 1],
            ]
        )

        self.inv_transform = np.linalg.inv(self.transform)

    def pixel_to_world(self, pixel_coord, depth):
        """
        Translate pixel coordinates to world coordinates.
        """
        x, y = pixel_coord
        pixel = np.matrix([[x], [y], [depth], [0]])
        return self.transform @ pixel

    def pixels_to_world(self, pixel_coords, depths):
        """
        Translate multiple pixel coordinates to world coordinates.
        """
        x, y = pixel_coords
        pixels = np.matrix([x, y, depths, np.ones(len(depths))])
        print(pixels)
        return self.transform @ pixels

    def world_to_pixel(self, world_coord):
        """
        Translate world coordinates to pixel coordinates
        """
        x, y, z, w = world_coord
        world_pixel = np.matrix([[x], [y], [z], [w]])
        return self.inv_transform @ world_pixel

    def world_to_pixels(self, world_coords):
        """
        Translate world coordinates to pixel coordinates
        """
        x, y, z = world_coords
        world_pixel = np.matrix([x, y, z, np.zeros(len(x))])
        return self.inv_transform @ world_pixel


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
        self.focal_x = (x / 2) / np.tan(radians(field_of_view / 2))
        self.focal_y = (y / 2) / np.tan(radians(field_of_view / 2))
        self.cam_origin = cam_origin
        cam_x, cam_y, cam_z = cam_origin
        self.alpha = skew_factor
        self.inv_transform = np.linalg.inv(
            np.matrix([[self.focal_x, self.alpha, cam_x], [0, self.focal_y, cam_y], [0, 0, 1]])
        )
        self.transform = np.matrix(
            [[self.focal_x, self.alpha, cam_x, 0], [0, self.focal_y, cam_y, 0], [0, 0, 1, 0]]
        )

    def __eq__(self, other):
        return (
            isinstance(other, PinholeLens)
            and self.fov == other.fov
            and self.focal_x == other.focal_x
            and self.focal_y == other.focal_y
            and self.cam_origin == other.cam_origin
            and self.alpha == other.alpha
            and (self.inv_transform == other.inv_transform).all()
            and (self.transform == other.transform).all()
        )

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
        x, y = pixel_coords
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
