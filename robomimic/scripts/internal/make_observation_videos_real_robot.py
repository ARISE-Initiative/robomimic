"""
A convenience script to visualize image observations from batch dataset trajectories.
It will output a video that consists of observations from several trajectories in a batch.
"""

import os
import json
import h5py
import argparse
import imageio
import numpy as np
from tqdm import tqdm

# a list of 2-tuples per episode, used for hardcoding interventions - each should 
# correspond approximately to timestamps in seconds for intervention start / end
# INTERVENTION_SEGMENTS = [
#     [(3, 5), (9, 10)],
#     [],
#     [(8, 11)],
# ]
INTERVENTION_SEGMENTS = [
]

# exclude wrist images
EXCLUDE_WRIST = True

def filter_interventions(interventions, window):
    """
    Only keeps intervention signal if it is True for @window timesteps.
    """
    new_interventions = np.array([False for _ in range(len(interventions))])
    start_ind = -1
    consec_ints = 0
    for i in range(len(interventions)):
        if interventions[i]:
            if (i == 0 or  not interventions[i - 1]):
                # start of new sequence
                start_ind = i
            consec_ints += 1
        else:
            if consec_ints >= window:
                # end of sequence, and got enough consecutive ints
                new_interventions[start_ind: start_ind + consec_ints] = True
            # reset counting
            start_ind = -1
            consec_ints = 0

    if consec_ints >= window:
        # end of sequence, and got enough consecutive ints
        new_interventions[start_ind: start_ind + consec_ints] = True

    return new_interventions

def video_name_from_ep(video_name, ep_count):
    lst = video_name.split(".")
    return ".".join(lst[:-1]) + "_{}.".format(ep_count) + lst[-1]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch",
        type=str,
    )
    parser.add_argument(
        "--video_name", # name of video
        type=str,
    )
    parser.add_argument(
        "--n", # number of trajectories to dump to video
        type=int,
        default=None,
    )
    parser.add_argument(
        "--filter_key",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--batch_no_self_im",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--per_episode", # flag for per episode videos
        action='store_true',
    )
    parser.add_argument(
        "--video_skip",  # number of frames to subsample
        type=int,
        default=5,
    )
    parser.add_argument(
        "--video_fps",  # fps for video
        type=int,
        default=20,
    )
    parser.add_argument(
        "--int_filter_window",  # number of frames to subsample
        type=int,
        default=None,
    )
    args = parser.parse_args()

    np.random.seed(0)

    demo_path = os.path.dirname(args.batch)

    f = h5py.File(args.batch, "r")
    f2 = None
    if args.batch_no_self_im is not None:
        f2 = h5py.File(args.batch_no_self_im, "r")

    # list of all demonstrations episodes (sorted in increasing number order)
    if args.filter_key is not None:
        print("using filter key: {}".format(args.filter_key))
        demos = sorted([elem.decode("utf-8") for elem in np.array(f["mask/{}".format(args.filter_key)])])
    else:
        demos = list(f["data"].keys())
        inds = np.argsort([int(elem[5:]) for elem in demos])
        demos = [demos[i] for i in inds]
    if args.n is not None:
        demos = demos[:args.n]

    # create a video writer with imageio
    video_name = args.video_name
    if video_name is None:
        video_name = "./playback_{}.mp4".format(env_name)
    if not args.per_episode:
        video_writer = imageio.get_writer(video_name, fps=args.video_fps)
    video_skip = args.video_skip
    video_count = 0

    # filtering for intervention signal
    int_filter_window = args.int_filter_window

    # convert intervention segment timestamps (in seconds) to approximate frame numbers
    # (assumes that video skip is consistent between reference videos and new videos)
    CONVERTED_INTERVENTION_SEGMENTS = []
    for lst in INTERVENTION_SEGMENTS:
        lst1 = []
        for tup in lst:
            assert len(tup) == 2
            tup1 = (round(tup[0] * args.video_fps), round(tup[1] * args.video_fps))
            lst1.append(tup1)
        CONVERTED_INTERVENTION_SEGMENTS.append(lst1)

    # iterate through dataset trajectories
    for ep_ind in tqdm(range(len(demos))):

        # maybe make new video writer
        if args.per_episode:
            ep_video_name = video_name_from_ep(video_name, ep_count=ep_ind)
            video_writer = imageio.get_writer(ep_video_name, fps=args.video_fps)

        # episode name
        ep = demos[ep_ind]

        # get name of all image observations
        image_keys = [k for k in f["data/{}/obs".format(ep)] if "rgb" in k]
        assert len(image_keys) > 0
        num_images = f["data/{}/obs/{}".format(ep, image_keys[0])].shape[0]

        use_interventions = False
        if (f2 is not None) or (len(CONVERTED_INTERVENTION_SEGMENTS) > 0):
            # infer intervention signal first
            use_interventions = True
            if f2 is not None:
                interventions = []
                j = 0
                num_images2 = f2["data/{}/obs/{}".format(ep, image_keys[0])].shape[0]
                for i in range(num_images):
                    # check for equal observations
                    if j < num_images2 and np.array_equal(f["data/{}/obs/{}".format(ep, image_keys[0])][i], f2["data/{}/obs/{}".format(ep, image_keys[0])][j]):
                    # if j < f2["data/{}/actions".format(ep)].shape[0] and np.equal(f["data/{}/actions".format(ep)][i], f2["data/{}/actions".format(ep)][j]):
                        j += 1
                        interventions.append(True)
                    else:
                        interventions.append(False)
            else:
                # each tuple specifies an intervention interval
                interventions = np.array([False for _ in range(num_images)])
                for tup in CONVERTED_INTERVENTION_SEGMENTS[ep_ind]:
                    interventions[tup[0] : tup[1]] = True

        if int_filter_window is not None:
            # LPF on interventions
            interventions = filter_interventions(interventions=interventions, window=int_filter_window)

        # iterate through observations in this trajectory
        for i in range(num_images):
            # subsample frames
            if i % video_skip == 0:
                # concatenate image obs together
                if EXCLUDE_WRIST:
                    im = f["data/{}/obs/{}".format(ep, image_keys[0])][i]
                else:
                    im = [f["data/{}/obs/{}".format(ep, k)][i] for k in image_keys]
                    im = np.concatenate(im, axis=1)

                # maybe add a red border to indicate intervention
                if use_interventions and interventions[i]:
                    border_size = 5
                    im[:border_size, :, :] = [255., 0., 0.]
                    im[-border_size:, :, :] = [255., 0., 0.]
                    im[:, :border_size, :] = [255., 0., 0.]
                    im[:, -border_size:, :] = [255., 0., 0.]

                video_writer.append_data(im)

        # maybe close video writer
        if args.per_episode:
            video_writer.close()

    f.close()
    if not args.per_episode:
        video_writer.close()
    if f2 is not None:
        f2.close()
