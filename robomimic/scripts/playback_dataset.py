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

    # visualize depth too
    python playback_dataset.py --dataset /path/to/dataset.hdf5 \
        --use-obs --render_image_names agentview_image \
        --render_depth_names agentview_depth \
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
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

import robomimic
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
from robomimic.envs.env_base import EnvBase, EnvType

try:
    import mimicgen
except ImportError:
    print("WARNING: could not import mimicgen envs")


# Define default cameras to use for each env type
DEFAULT_CAMERAS = {
    EnvType.ROBOSUITE_TYPE: ["agentview"],
    EnvType.IG_MOMART_TYPE: ["rgb"],
    EnvType.GYM_TYPE: ValueError("No camera names supported for gym type env!"),
    EnvType.REAL_TYPE: ["front_image"],
    EnvType.GPRS_REAL_TYPE: ["front_image"],
}


def add_red_border(frame):
    """Add a red border to image frame."""
    border_size = int(0.05 * min(frame.shape[0], frame.shape[1])) # 5% of image
    frame[:border_size, :, :] = [255., 0., 0.]
    frame[-border_size:, :, :] = [255., 0., 0.]
    frame[:, :border_size, :] = [255., 0., 0.]
    frame[:, -border_size:, :] = [255., 0., 0.]
    return frame


def depth_to_rgb(depth_map, depth_min=None, depth_max=None):
    """
    Convert depth map to rgb array by computing normalized depth values in [0, 1].
    """
    # normalize depth map into [0, 1]
    if depth_min is None:
        depth_min = depth_map.min()
    if depth_max is None:
        depth_max = depth_map.max()
    depth_map = (depth_map - depth_min) / (depth_max - depth_min)
    # depth_map = np.clip(depth_map / 3., 0., 1.)
    if len(depth_map.shape) == 3:
        assert depth_map.shape[-1] == 1
        depth_map = depth_map[..., 0]
    assert len(depth_map.shape) == 2 # [H, W]
    return (255. * cm.hot(depth_map, 3)).astype(np.uint8)[..., :3]


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
    interventions=None,
    real=False,
):
    """
    Helper function to playback a single trajectory using the simulator environment.
    If @actions are not None, it will play them open-loop after loading the initial state. 
    Otherwise, @states are loaded one by one.

    Args:
        env (instance of EnvBase): environment
        initial_state (dict): initial simulation state to load
        states (list of dict or np.array): array of simulation states to load
        actions (np.array): if provided, play actions back open-loop instead of using @states
        render (bool): if True, render on-screen
        video_writer (imageio writer): video writer
        video_skip (int): determines rate at which environment frames are written to video
        camera_names (list): determines which camera(s) are used for rendering. Pass more than
            one to output a video with multiple camera views concatenated horizontally.
        first (bool): if True, only use the first frame of each episode.
        real (bool): if True, playback is happening on real robot
    """
    assert isinstance(env, EnvBase)

    write_video = (video_writer is not None)
    video_count = 0
    assert not (render and write_video)

    # load the initial state
    env.reset()
    if real:
        assert actions is not None, "must supply actions for real robot playback"
        traj_len = actions.shape[0]
        input("ready for next episode? hit enter to continue")
    else:
        env.reset_to(initial_state)
        traj_len = len(states)

    action_playback = (actions is not None)
    if action_playback:
        assert len(states) == actions.shape[0]

    for i in range(traj_len):
        if action_playback:
            env.step(actions[i])
            if (i < traj_len - 1) and not real:
                # check whether the actions deterministically lead to the same recorded states
                state_playback = env.get_state()["states"]
                if isinstance(state_playback, dict):
                    # state is dict, so assert equality for all keys
                    for k in state_playback:
                        if not np.all(np.equal(states[i + 1][k], state_playback[k])):
                            err = np.linalg.norm(states[i + 1][k] - state_playback[k])
                            print("warning: playback diverged by {} at step {} state key {}".format(err, i, k))
                else:
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
                    frame = env.render(mode="rgb_array", height=512, width=512, camera_name=cam_name)
                    if (interventions is not None) and interventions[i]:
                        # add red border to frame
                        frame = add_red_border(frame=frame)
                    video_img.append(frame)
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
    depth_names=None,
    first=False,
    intervention=False,
):
    """
    This function reads all "rgb" observations in the dataset trajectory and
    writes them into a video.

    Args:
        traj_grp (hdf5 file group): hdf5 group which corresponds to the dataset trajectory to playback
        video_writer (imageio writer): video writer
        video_skip (int): determines rate at which environment frames are written to video
        image_names (list): determines which image observations are used for rendering. Pass more than
            one to output a video with multiple image observations concatenated horizontally.
        depth_names (list): determines which depth observations are used for rendering (if any).
        first (bool): if True, only use the first frame of each episode.
        intervention (bool): if True, denote intervention timesteps with a red border
    """
    assert image_names is not None, "error: must specify at least one image observation to use in @image_names"
    video_count = 0

    # figure out which frame indices to iterate over
    traj_len = traj_grp["actions"].shape[0]
    frame_inds = range(traj_len)
    if first:
        video_skip = 1 # keep all frames
        if intervention:
            # find where interventions begin (0 to 1 edge) and get frames right before them
            if len(traj_grp["interventions"].shape) == 2:
                all_interventions = traj_grp["interventions"][:, 0].astype(int)
            else:
                all_interventions = traj_grp["interventions"][:].astype(int)
            frame_inds = list(np.nonzero((all_interventions[1:] - all_interventions[:-1]) > 0)[0])
        else:
            frame_inds = range(1)

    if depth_names is not None:
        # compute min and max depth value across trajectory for normalization
        depth_min = { k : traj_grp["obs/{}".format(k)][:].min() for k in depth_names }
        depth_max = { k : traj_grp["obs/{}".format(k)][:].max() for k in depth_names }

    for i in frame_inds:
        if video_count % video_skip == 0:
            # concatenate image obs together
            im = [traj_grp["obs/{}".format(k)][i] for k in image_names]
            depth = [depth_to_rgb(traj_grp["obs/{}".format(k)][i], depth_min=depth_min[k], depth_max=depth_max[k]) for k in depth_names] if depth_names is not None else []
            frame = np.concatenate(im + depth, axis=1)
            if intervention and traj_grp["interventions"][i]:
                # add red border to frame
                frame = add_red_border(frame=frame)
            video_writer.append_data(frame)
        video_count += 1


def playback_dataset(args, env=None):
    # some arg checking
    write_video = (args.video_path is not None)
    assert not (args.render and write_video) # either on-screen or video but not both
    if args.absolute:
        assert args.use_actions

    # Auto-fill camera rendering info if not specified
    if args.render_image_names is None:
        # We fill in the automatic values
        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=args.dataset)
        env_type = EnvUtils.get_env_type(env_meta=env_meta)
        args.render_image_names = DEFAULT_CAMERAS[env_type]

    if args.render:
        # on-screen rendering can only support one camera
        assert len(args.render_image_names) == 1

    if args.use_obs:
        assert write_video, "playback with observations can only write to video"
        assert not args.use_actions, "playback with observations is offline and does not support action playback"

    if args.render_depth_names is not None:
        assert args.use_obs, "depth observations can only be visualized from observations currently"

    # create environment only if not playing back with observations
    if not args.use_obs:
        # need to make sure ObsUtils knows which observations are images, but it doesn't matter 
        # for playback since observations are unused. Pass a dummy spec here.
        dummy_spec = dict(
            obs=dict(
                    low_dim=["robot0_eef_pos"],
                    rgb=[],
                ),
        )

        # some operations for playback are env-type-specific
        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=args.dataset)
        is_robosuite_env = EnvUtils.is_robosuite_env(env_meta)
        is_real_robot = EnvUtils.is_real_robot_env(env_meta) or EnvUtils.is_real_robot_gprs_env(env_meta)

        if args.absolute:
            # modify env-meta to tell the environment to expect absolute actions
            assert is_robosuite_env or is_real_robot, "only these support absolute actions for now"
            if is_robosuite_env:
                env_meta["env_kwargs"]["controller_configs"]["control_delta"] = False
            else:
                env_meta["env_kwargs"]["absolute_actions"] = True

        if env is None:
            if is_real_robot:
                # TODO: update hardcoded keys on real robot
                dummy_spec["obs"]["rgb"] = ["front_image", "wrist_image", "side_image"]
                dummy_spec["obs"]["depth"] = ["front_image_depth", "wrist_image_depth", "side_image_depth"]
            ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs=dummy_spec)
            env = EnvUtils.create_env_from_metadata(env_meta=env_meta, render=args.render, render_offscreen=write_video)

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
        fps = 5 if args.first else 20
        video_writer = imageio.get_writer(args.video_path, fps=fps)

    for ind in range(len(demos)):
        ep = demos[ind]
        print("Playing back episode: {}".format(ep))

        if args.use_obs:
            playback_trajectory_with_obs(
                traj_grp=f["data/{}".format(ep)], 
                video_writer=video_writer, 
                video_skip=args.video_skip,
                image_names=args.render_image_names,
                depth_names=args.render_depth_names,
                first=args.first,
                intervention=args.intervention,
            )
            continue

        # prepare states to reload from
        if not is_real_robot:
            states = f["data/{}/states".format(ep)][()]
            initial_state = dict(states=states[0])
            if is_robosuite_env:
                initial_state["model"] = f["data/{}".format(ep)].attrs["model_file"]

        # supply actions if using open-loop action playback
        actions = None
        if args.use_actions:
            if args.absolute:
                actions = f["data/{}/actions_abs".format(ep)][()]
            else:
                actions = f["data/{}/actions".format(ep)][()]

        if is_real_robot:
            assert actions is not None
            states = np.zeros(actions.shape[0])
            initial_state = dict(states=states[0])
            
        # supply interventions if we need them for visualization
        interventions = None
        if args.intervention:
            interventions = f["data/{}/interventions".format(ep)][()]

        playback_trajectory_with_env(
            env=env, 
            initial_state=initial_state, 
            states=states, actions=actions, 
            render=args.render, 
            video_writer=video_writer, 
            video_skip=args.video_skip,
            camera_names=args.render_image_names,
            first=args.first,
            interventions=interventions,
            real=is_real_robot,
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

    # TODO: clean up this arg
    parser.add_argument(
        "--absolute",
        action='store_true',
        help="use absolute actions for open-loop action playback",
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
        default=None,
        help="(optional) camera name(s) / image observation(s) to use for rendering on-screen or to video. Default is"
             "None, which corresponds to a predefined camera for each env type",
    )

    parser.add_argument(
        "--render_depth_names",
        type=str,
        nargs='+',
        default=None,
        help="(optional) depth observation(s) to use for rendering to video"
    )

    # Only use the first frame of each episode
    parser.add_argument(
        "--first",
        action='store_true',
        help="use first frame of each episode",
    )

    # Denote intervention timesteps with a red border in the frame
    parser.add_argument(
        "--intervention",
        action='store_true',
        help="denote intervention timesteps with a red border in the frame",
    )

    args = parser.parse_args()
    playback_dataset(args)
