"""
Script to generate gallery image / video.
"""
import os
import json
import imageio
import numpy as np
from tqdm import tqdm
from PIL import Image

BASE_VIDEO_DIR = "/tmp/videos"

VIDEO_PATHS = [
    os.path.join(BASE_VIDEO_DIR, "playback_lift_se.mp4"),
    os.path.join(BASE_VIDEO_DIR, "playback_can_se.mp4"),
    os.path.join(BASE_VIDEO_DIR, "playback_square_se.mp4"),
    os.path.join(BASE_VIDEO_DIR, "playback_transport_se.mp4"),
    os.path.join(BASE_VIDEO_DIR, "playback_tool_hang_se.mp4"),

    os.path.join(BASE_VIDEO_DIR, "playback_lift_mh_better_1.mp4"),
    os.path.join(BASE_VIDEO_DIR, "playback_can_mh_better_1.mp4"),
    os.path.join(BASE_VIDEO_DIR, "playback_square_mh_better_1.mp4"),
    os.path.join(BASE_VIDEO_DIR, "playback_transport_mh_better.mp4"),

    os.path.join(BASE_VIDEO_DIR, "playback_lift_mh_better_2.mp4"),
    os.path.join(BASE_VIDEO_DIR, "playback_can_mh_better_2.mp4"),
    os.path.join(BASE_VIDEO_DIR, "playback_square_mh_better_2.mp4"),
    os.path.join(BASE_VIDEO_DIR, "playback_transport_mh_okay_better.mp4"),

    os.path.join(BASE_VIDEO_DIR, "playback_lift_mh_okay_1.mp4"),
    os.path.join(BASE_VIDEO_DIR, "playback_can_mh_okay_1.mp4"),
    os.path.join(BASE_VIDEO_DIR, "playback_square_mh_okay_1.mp4"),
    os.path.join(BASE_VIDEO_DIR, "playback_transport_mh_okay.mp4"),

    os.path.join(BASE_VIDEO_DIR, "playback_lift_mh_okay_2.mp4"),
    os.path.join(BASE_VIDEO_DIR, "playback_can_mh_okay_2.mp4"),
    os.path.join(BASE_VIDEO_DIR, "playback_square_mh_okay_2.mp4"),
    os.path.join(BASE_VIDEO_DIR, "playback_transport_mh_worse_better.mp4"),

    os.path.join(BASE_VIDEO_DIR, "playback_lift_mh_worse_1.mp4"),
    os.path.join(BASE_VIDEO_DIR, "playback_can_mh_worse_1.mp4"),
    os.path.join(BASE_VIDEO_DIR, "playback_square_mh_worse_1.mp4"),
    os.path.join(BASE_VIDEO_DIR, "playback_transport_mh_worse_okay.mp4"),

    os.path.join(BASE_VIDEO_DIR, "playback_lift_mh_worse_2.mp4"),
    os.path.join(BASE_VIDEO_DIR, "playback_can_mh_worse_2.mp4"),
    os.path.join(BASE_VIDEO_DIR, "playback_square_mh_worse_2.mp4"),
    os.path.join(BASE_VIDEO_DIR, "playback_transport_mh_worse.mp4"),

    os.path.join(BASE_VIDEO_DIR, "playback_liftreal.mp4"),
    os.path.join(BASE_VIDEO_DIR, "playback_canreal.mp4"),
    os.path.join(BASE_VIDEO_DIR, "playback_toolhangreal.mp4"),
]

# expected size of video frames - if mismatch, will be reshaped
VIDEO_FRAME_SIZE = (512, 512)


def env_name_from_filename(filename):
    """
    Assumes that format of name is playback_{env}_...
    """
    return filename.split("_")[1]


def get_video_length(path):
    """
    Helper function to get length of video.
    """
    nframes = 0
    reader = imageio.get_reader(path)
    nframes = reader.count_frames()
    reader.close()
    return nframes


def read_video_frames(path, start, nframes):
    """
    Helper function to read @nframes frames from a video,
    starting with frame number @start.
    """
    frames = []
    reader = imageio.get_reader(path)
    reader.set_image_index(start)
    for _ in range(nframes):
        frames.append(reader.get_next_data())
    reader.close()
    return frames


def build_env_to_video_map():
    """
    Builds a dictionary from env_name to list of videos for that env, and
    corresponding video lengths.
    """
    print("building metadata for generation...")
    env_map = dict()
    for path in tqdm(VIDEO_PATHS):
        fname = os.path.basename(path)
        env_name = env_name_from_filename(fname)
        if env_name not in env_map:
            env_map[env_name] = dict(video_paths=[], video_lengths=[])
        env_map[env_name]["video_paths"].append(path)
        env_map[env_name]["video_lengths"].append(get_video_length(path))
    return env_map


def sample_gallery(num_rows, num_cols, num_images):
    """
    Helper function to handle sampling video paths and frame indices
    for the gallery.
    """
    env_map = build_env_to_video_map()
    sampled_video_paths = [[None for _ in range(num_cols)] for _ in range(num_rows)]
    sampled_frame_inds = [[None for _ in range(num_cols)] for _ in range(num_rows)]
    envs = list(env_map.keys())
    num_envs = len(envs)
    prev_env = None
    for r_ind in range(num_rows):
        for c_ind in range(num_cols):
            # make sure we get a unique env from last time
            env = prev_env
            while env == prev_env:
                env_ind = np.random.randint(num_envs)
                env = envs[env_ind]
            video_ind = np.random.randint(len(env_map[env]["video_paths"]))
            video_path = env_map[env]["video_paths"][video_ind]
            video_length = env_map[env]["video_lengths"][video_ind]
            random_frame_ind = np.random.randint(video_length - num_images + 1)

            sampled_video_paths[r_ind][c_ind] = video_path
            sampled_frame_inds[r_ind][c_ind] = random_frame_ind
            prev_env = env

    return sampled_video_paths, sampled_frame_inds


def generate_gallery(num_rows, num_cols, num_images):
    """
    Samples the frames for the gallery image, or a sequence of frames if 
    @num_images is greater than 1 (the frames will be sequential from
    each video). Writes to an image if @num_images is 1, otherwise a video.
    """

    # sample the video paths and frame indices to use for the gallery grid
    sampled_video_paths, sampled_frame_inds = sample_gallery(num_rows, num_cols, num_images)

    write_video = (num_images > 1)
    if write_video:
        video_writer = imageio.get_writer("/tmp/test.mp4", fps=20)

    for video_ind in range(num_images):
        print("generating image {}...".format(video_ind))
        all_frames = []
        prev_env = None
        for r_ind in tqdm(range(num_rows)):
            row_image = []
            for c_ind in range(num_cols):
                # read the correct frame from the appropriate video
                frame = read_video_frames(
                    path=sampled_video_paths[r_ind][c_ind], 
                    start=sampled_frame_inds[r_ind][c_ind] + video_ind, 
                    nframes=1,
                )[0]

                # maybe resize video frame
                if (frame.shape[0] != VIDEO_FRAME_SIZE[0]) or (frame.shape[1] != VIDEO_FRAME_SIZE[1]):
                    frame = Image.fromarray(frame)
                    frame = frame.resize((VIDEO_FRAME_SIZE[1], VIDEO_FRAME_SIZE[0]))
                    frame = np.asarray(frame)

                row_image.append(frame)
            # concatenate images into a row
            row_image = np.concatenate(row_image, axis=1)
            all_frames.append(row_image)

        # concatenate row images into a single image
        all_frames = np.concatenate(all_frames, axis=0)

        image = Image.fromarray(all_frames)
        image = image.resize((200 * num_cols, 200 * num_rows))
        if write_video:
            image = np.asarray(image)
            video_writer.append_data(image)
        else:
            image.save("/tmp/test.png") 

    if write_video:
        video_writer.close()


if __name__ == "__main__":
    # generate_gallery(num_rows=9, num_cols=8, num_images=1)
    # generate_gallery(num_rows=4, num_cols=8, num_images=100)
    generate_gallery(num_rows=4, num_cols=8, num_images=1)
    # env_map = build_env_to_video_map()
    # print(json.dumps(env_map, indent=4))

