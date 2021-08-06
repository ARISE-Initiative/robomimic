"""
A script to visualize dataset trajectories by loading the simulation states
one by one or loading the first state and playing actions back open-loop.
The script can generate videos as well, by rendering simulation frames
during playback. The videos can also be generated using the image observations
in the dataset (this is useful for real-robot datasets) by using the
--use-obs argument.

Args:
    dataset (str): path to hdf5 dataset

    filter_key (str): if provided, use the subset of trajectories
        in the file that correspond to this filter key

    n (int): if provided, stop after n trajectories are processed

    use-obs (bool): if flag is provided, visualize trajectories with dataset 
        image observations instead of simulator

    use-actions (bool): if flag is provided, use open-loop action playback 
        instead of loading sim states

    render (bool): if flag is provided, use on-screen rendering during playback
    
    video_path (str): if provided, render trajectories to this video file path

    video_skip (int): render frames to a video every @video_skip steps

    render_image_names (str or [str]): camera name(s) / image observation(s) to 
        use for rendering on-screen or to video

    first (bool): if flag is provided, use first frame of each episode for playback
        instead of the entire episode. Useful for visualizing task initializations.

Example usage below:

    # force simulation states one by one, and render agentview and wrist view cameras to video
    python playback_dataset.py --dataset /path/to/dataset.hdf5 \
        --render_image_names agentview robot0_eye_in_hand \
        --video_path /tmp/playback_dataset.mp4

    # playback the actions in the dataset, and render agentview camera during playback to video
    python playback_dataset.py --dataset /path/to/dataset.hdf5 \
        --use-actions --render_image_names agentview \
        --video_path /tmp/playback_dataset_with_actions.mp4

    # use the observations stored in the dataset to render videos of the dataset trajectories
    python playback_dataset.py --dataset /path/to/dataset.hdf5 \
        --use-obs --render_image_names agentview_image \
        --video_path /tmp/obs_trajectory.mp4

    # visualize initial states in the demonstration data
    python playback_dataset.py --dataset /path/to/dataset.hdf5 \
        --first --render_image_names agentview \
        --video_path /tmp/dataset_task_inits.mp4
"""

import os
import json
import h5py
import argparse
import imageio
import numpy as np

import robomimic
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
from robomimic.envs.env_base import EnvBase


def playback_trajectory_with_env(
    env, 
    initial_state, 
    states, 
    actions=None, 
    render=False, 
    video_writer=None, 
    video_skip=5, 
    camera_names=None,
    first=False,
):
    """
    Helper function to playback a single trajectory using the simulator environment.
    If @actions are not None, it will play them open-loop after loading the initial state. 
    Otherwise, @states are loaded one by one.

    Args:
        env (instance of EnvBase): environment
        initial_state (dict): initial simulation state to load
        states (np.array): array of simulation states to load
        actions (np.array): if provided, play actions back open-loop instead of using @states
        render (bool): if True, render on-screen
        video_writer (imageio writer): video writer
        video_skip (int): determines rate at which environment frames are written to video
        camera_names (list): determines which camera(s) are used for rendering. Pass more than
            one to output a video with multiple camera views concatenated horizontally.
        first (bool): if True, only use the first frame of each episode.
    """
    assert isinstance(env, EnvBase)

    write_video = (video_writer is not None)
    video_count = 0
    assert not (render and write_video)

    # load the initial state
    env.reset()
    env.reset_to(initial_state)

    traj_len = states.shape[0]
    action_playback = (actions is not None)
    if action_playback:
        assert states.shape[0] == actions.shape[0]

    for i in range(traj_len):
        if action_playback:
            env.step(actions[i])
            if i < traj_len - 1:
                # check whether the actions deterministically lead to the same recorded states
                state_playback = env.get_state()["states"]
                if not np.all(np.equal(states[i + 1], state_playback)):
                    err = np.linalg.norm(states[i + 1] - state_playback)
                    print("warning: playback diverged by {} at step {}".format(err, i))
        else:
            env.reset_to({"states" : states[i]})

        # on-screen render
        if render:
            env.render(mode="human", camera_name=camera_names[0])

        # video render
        if write_video:
            if video_count % video_skip == 0:
                video_img = []
                for cam_name in camera_names:
                    video_img.append(env.render(mode="rgb_array", height=512, width=512, camera_name=cam_name))
                video_img = np.concatenate(video_img, axis=1) # concatenate horizontally
                video_writer.append_data(video_img)
            video_count += 1

        if first:
            break


def playback_trajectory_with_obs(
    traj_grp,
    video_writer, 
    video_skip=5, 
    image_names=None,
    first=False,
):
    """
    This function reads all "image" observations in the dataset trajectory and
    writes them into a video.

    Args:
        traj_grp (hdf5 file group): hdf5 group which corresponds to the dataset trajectory to playback
        video_writer (imageio writer): video writer
        video_skip (int): determines rate at which environment frames are written to video
        image_names (list): determines which image observations are used for rendering. Pass more than
            one to output a video with multiple image observations concatenated horizontally.
        first (bool): if True, only use the first frame of each episode.
    """
    assert image_names is not None, "error: must specify at least one image observation to use in @image_names"
    video_count = 0

    traj_len = traj_grp["actions"].shape[0]
    for i in range(traj_len):
        if video_count % video_skip == 0:
            # concatenate image obs together
            im = [traj_grp["obs/{}".format(k)][i] for k in image_names]
            frame = np.concatenate(im, axis=1)
            video_writer.append_data(frame)
        video_count += 1

        if first:
            break


def playback_dataset(args):
    # some arg checking
    write_video = (args.video_path is not None)
    assert not (args.render and write_video) # either on-screen or video but not both
    if args.render:
        # on-screen rendering can only support one camera
        assert len(args.render_image_names) == 1
    if args.use_obs:
        assert write_video, "playback with observations can only write to video"
        assert not args.use_actions, "playback with observations is offline and does not support action playback"

    # create environment only if not playing back with observations
    if not args.use_obs:
        # need to make sure ObsUtils knows which observations are images, but it doesn't matter 
        # for playback since observations are unused. Pass a dummy spec here.
        dummy_spec = dict(
            obs=dict(
                    low_dim=["robot0_eef_pos"],
                    image=[],
                ),
        )
        ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs=dummy_spec)

        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=args.dataset)
        env = EnvUtils.create_env_from_metadata(env_meta=env_meta, render=args.render, render_offscreen=write_video)

        # some operations for playback are robosuite-specific, so determine if this environment is a robosuite env
        is_robosuite_env = EnvUtils.is_robosuite_env(env_meta)

    f = h5py.File(args.dataset, "r")

    # list of all demonstration episodes (sorted in increasing number order)
    if args.filter_key is not None:
        print("using filter key: {}".format(args.filter_key))
        demos = [elem.decode("utf-8") for elem in np.array(f["mask/{}".format(args.filter_key)])]
    else:
        demos = list(f["data"].keys())
    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]

    # maybe reduce the number of demonstrations to playback
    if args.n is not None:
        demos = demos[:args.n]

    # maybe dump video
    video_writer = None
    if write_video:
        video_writer = imageio.get_writer(args.video_path, fps=20)

    for ind in range(len(demos)):
        ep = demos[ind]
        print("Playing back episode: {}".format(ep))

        if args.use_obs:
            playback_trajectory_with_obs(
                traj_grp=f["data/{}".format(ep)], 
                video_writer=video_writer, 
                video_skip=args.video_skip,
                image_names=args.render_image_names,
                first=args.first,
            )
            continue

        # prepare initial state to reload from
        states = f["data/{}/states".format(ep)][()]
        initial_state = dict(states=states[0])
        if is_robosuite_env:
            initial_state["model"] = f["data/{}".format(ep)].attrs["model_file"]

        # supply actions if using open-loop action playback
        actions = None
        if args.use_actions:
            actions = f["data/{}/actions".format(ep)][()]

        playback_trajectory_with_env(
            env=env, 
            initial_state=initial_state, 
            states=states, actions=actions, 
            render=args.render, 
            video_writer=video_writer, 
            video_skip=args.video_skip,
            camera_names=args.render_image_names,
            first=args.first,
        )

    f.close()
    if write_video:
        video_writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        help="path to hdf5 dataset",
    )
    parser.add_argument(
        "--filter_key",
        type=str,
        default=None,
        help="(optional) filter key, to select a subset of trajectories in the file",
    )

    # number of trajectories to playback. If omitted, playback all of them.
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="(optional) stop after n trajectories are played",
    )

    # Use image observations instead of doing playback using the simulator env.
    parser.add_argument(
        "--use-obs",
        action='store_true',
        help="visualize trajectories with dataset image observations instead of simulator",
    )

    # Playback stored dataset actions open-loop instead of loading from simulation states.
    parser.add_argument(
        "--use-actions",
        action='store_true',
        help="use open-loop action playback instead of loading sim states",
    )

    # Whether to render playback to screen
    parser.add_argument(
        "--render",
        action='store_true',
        help="on-screen rendering",
    )

    # Dump a video of the dataset playback to the specified path
    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="(optional) render trajectories to this video file path",
    )

    # How often to write video frames during the playback
    parser.add_argument(
        "--video_skip",
        type=int,
        default=5,
        help="render frames to video every n steps",
    )

    # camera names to render, or image observations to use for writing to video
    parser.add_argument(
        "--render_image_names",
        type=str,
        nargs='+',
        default=["agentview"],
        help="(optional) camera name(s) / image observation(s) to use for rendering on-screen or to video",
    )

    # Only use the first frame of each episode
    parser.add_argument(
        "--first",
        action='store_true',
        help="use first frame of each episode",
    )

    args = parser.parse_args()
    playback_dataset(args)
