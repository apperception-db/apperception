### IMPORTS
import os
import sys

from numpy.core.fromnumeric import var
sys.path.append(os.path.join(os.getcwd(),"apperception"))

### IMPORTS
import cv2

from world import *
from world_executor import *
from video_util import *
from metadata_util import *
import lens

import psycopg2

#import tasm
def plot_3d(object_bboxes):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from itertools import product, combinations
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for bbox in object_bboxes:
        x_min, y_min, z_min, x_max, y_max, z_max = bbox
        points = np.array([[x_min, y_min, z_min], [x_min, y_min, z_max], [x_min, y_max, z_min], [x_min, y_max, z_max],
                  [x_max, y_min, z_min], [x_max, y_min, z_max], [x_max, y_max, z_min], [x_max, y_max, z_max]])
        vertices = [(points[0], points[4]), 
                    (points[0], points[2]), 
                    (points[0], points[1]),
                    (points[1], points[3]), 
                    (points[1], points[5]), 
                    (points[3], points[7]),
                    (points[3], points[2]),
                    (points[4], points[5]),
                    (points[5], points[7]),
                    (points[2], points[6]),
                    (points[6], points[7]),
                    (points[4], points[6])]
        for s, e in vertices:
            ax.plot3D(*zip(s, e), color='b')
            
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()
    
### Let's define some attribute for constructing the world first
name = 'traffic_scene' # world name
units = 'metrics'      # world units
cam1_id = 'cam1'
# cam1_video_file = "../../visual_road_video_cut/single_car_000.mp4" #cam1 view
cam1_video_file = "../../visual_road_video_cut/traffic-001_clip_clip.mp4" #cam1 view
cam2_id = 'cam2'
# cam2_video_file = "../../visual_road_video_cut/single_car_001.mp4" #cam2 view
cam2_video_file = "../../visual_road_video_cut/traffic-002_clip_clip.mp4" #cam2 view

### First we define a world
traffic_world = World(name=name, units=units)

### Secondly we construct the camera
# cam1_len = lens.VRLens(resolution=[960, 540], cam_origin=(202, -242,  1), yaw=135, roll=0, pitch=0, field_of_view=90)
# cam2_len = lens.VRLens(resolution=[960, 540], cam_origin=(210, -270, 9), yaw=225, roll=0, pitch=0, field_of_view=90)
cam1_len = lens.VRLens(resolution=[3840, 2160], cam_origin=(199, -239, 5.2), yaw=-52, roll=0, pitch=0, field_of_view=90)
cam2_len = lens.VRLens(resolution=[3840, 2160], cam_origin=(210, -270, 9), yaw=128, roll=0, pitch=-30, field_of_view=90)
fps = 30

### Ingest the camera to the world
traffic_world = traffic_world.camera(cam_id=cam1_id, 
                               video_file=cam1_video_file, 
                               metadata_identifier=name+"_"+cam1_id, 
                               lens=cam1_len)
traffic_world = traffic_world.camera(cam_id=cam2_id, 
                               video_file=cam2_video_file, 
                               metadata_identifier=name+"_"+cam2_id, 
                               lens=cam2_len)
# cam1_recognized_world = traffic_world.recognize(cam1_id).execute()
# merged_bbox_1 = traffic_world.predicate(lambda obj:obj.object_type == "car").get_merged_geo(distinct=True).execute()

# cam2_recognized_world = traffic_world.recognize(cam2_id).execute()
# merged_bbox_2 = traffic_world.predicate(lambda obj:obj.object_type == "car").get_merged_geo(distinct=True).execute()
# plot_3d(merged_bbox_2[0][0].T)
# import code; code.interact(local=vars())
# for merged_bbox in merged_bbox_2:
#     plot_3d(np.asarray([np.asarray(a) for  a in merged_bbox[0]]).T)
filtered_world = traffic_world.predicate(lambda obj:obj.object_type == "car")
returned_trajectory = filtered_world.get_trajectory().execute()
traffic_world.overlay_trajectory(cam1_id, returned_trajectory)
### Call execute on the world to run the detection algorithm and save the real data to the database
# recognized_world = traffic_world.recognize(cam1_id)
# recognized_world.execute()

# tracking_results = recognize(cam2_video_file)
# file1 = open("resultfile_2.txt","w")
# file1.write(str(tracking_results))
# file1.close()
# conn = psycopg2.connect(database="mobilitydb", user="docker", password="docker", host="localhost", port=25432)
# add_recognized_objs(conn, cam2_len, tracking_results, traffic_world.VideoContext.start_time, temp=True)

# volume = traffic_world.select_intersection_of_interest_or_use_default(cam_id=cam2_id)
# filtered_world = traffic_world.predicate(lambda obj:obj.object_type == "car")
# # filtered_world = filtered_world.interval([0,fps*3])
 
# ### to get the trajectory and the video over the entire trajectory(amber case)
# filtered_ids = filtered_world.selectkey(distinct = True).execute()
# print("filtered_ids are", filtered_ids)
# print(len(filtered_ids))
# if len(filtered_ids)>0:
#     id_array = [e[0] for e in filtered_ids]
#     ### Fetch the trajectory of these items
#     trajectory = traffic_world.predicate(lambda obj: obj.object_id in id_array, {"id_array":id_array}).get_trajectory(distinct=True).execute()
#     traffic_world.overlay_trajectory(cam2_id, trajectory)
#     ## Get the videos of these items
#     entire_video = traffic_world.predicate(lambda obj: obj.object_id in id_array, {"id_array":id_array}).get_video()
#     entire_video.execute()
