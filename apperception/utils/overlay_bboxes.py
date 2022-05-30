from apperception.video_util import (convert_datetime_to_frame_num,
                                     get_video_box, get_video_roi)
from apperception.legacy.world_executor import (create_transform_matrix,
                                         reformat_fetched_world_coords,
                                         world_to_pixel)


def overlay_bboxes(metadata_results, cams, boxed):
    # The cam nodes are raw data from the database
    # TODO: I forget why we used the data from the db instead of directly fetch
    # from the world

    video_files = []
    for cam in cams:
        cam_x, cam_y, focal_x, focal_y, skew_factor = (
            cam.lens.cam_origin[0],
            cam.lens.cam_origin[1],
            cam.lens.focal_x,
            cam.lens.focal_y,
            cam.lens.alpha,
        )
        start_time = cam.configs[0].timestamp
        cam_video_file = cam.video_file
        transform_matrix = create_transform_matrix(focal_x, focal_y, cam_x, cam_y, skew_factor)

        for item_id, vals in metadata_results.items():
            world_coords, timestamps = vals
            # print("timestamps are", timestamps)
            world_coords = reformat_fetched_world_coords(world_coords)
            cam_coords = world_to_pixel(world_coords, transform_matrix)

            vid_times = convert_datetime_to_frame_num(start_time, timestamps)
            # print(vid_times)

            vid_fname = "./output/" + cam.metadata_id + item_id + ".mp4"
            # print(vid_fname)
            if boxed:
                get_video_box(vid_fname, cam_video_file, cam_coords, vid_times)
            else:
                get_video_roi(vid_fname, cam_video_file, cam_coords, vid_times)
            video_files.append(vid_fname)
    print("output video files", ",".join(video_files))
    return video_files
