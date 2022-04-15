"""
Helper script to convert the RoboTurk Pilot datasets (https://roboturk.stanford.edu/dataset_sim.html)
into a format compatible with this repository. It will also create some useful filter keys
in the file (e.g. training, validation, and fastest n trajectories). Prior work
(https://arxiv.org/abs/1911.05321) has found this useful (for example, training on the 
fastest 225 demonstrations for bins-Can).

Direct download link for dataset: http://cvgl.stanford.edu/projects/roboturk/RoboTurkPilot.zip

Args:
    folder (str): path to a folder containing a demo.hdf5 and a models directory containing
        mujoco xml files. For example, RoboTurkPilot/bins-Can.

    n (int): creates a filter key corresponding to the n fastest trajectories. Defaults to 225.

Example usage:

    python convert_roboturk_pilot.py --folder /path/to/RoboTurkPilot/bins-Can --n 225
"""

import os
import h5py
import json
import argparse
import numpy as np
from tqdm import tqdm

import robomimic
import robomimic.envs.env_base as EB
from robomimic.utils.file_utils import create_hdf5_filter_key
from robomimic.scripts.split_train_val import split_train_val_from_hdf5


def convert_rt_pilot_hdf5(ref_folder):
    """
    Uses the reference demo hdf5 to write a new converted hdf5 compatible with
    the repository.

    Args:
        ref_folder (str): path to a folder containing a demo.hdf5 and a models directory containing
            mujoco xml files.
    """
    hdf5_path = os.path.join(ref_folder, "demo.hdf5")
    new_path = os.path.join(ref_folder, "demo_new.hdf5")

    f = h5py.File(hdf5_path, "r")
    f_new = h5py.File(new_path, "w")
    f_new_grp = f_new.create_group("data")

    # sorted list of demonstrations by demo number
    demos = list(f["data"].keys())
    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]

    # write each demo
    num_samples_arr = []
    for demo_id in tqdm(range(len(demos))):
        ep = demos[demo_id]

        # create group for this demonstration
        ep_data_grp = f_new_grp.create_group(ep)

        # copy states over
        states = f["data/{}/states".format(ep)][()]
        ep_data_grp.create_dataset("states", data=np.array(states))

        # concat jvels and gripper actions to form full actions
        jvels = f["data/{}/joint_velocities".format(ep)][()]
        gripper_acts = f["data/{}/gripper_actuations".format(ep)][()]
        actions = np.concatenate([jvels, gripper_acts], axis=1)

        # IMPORTANT: clip actions to -1, 1, since this is expected by the codebase
        actions = np.clip(actions, -1., 1.)
        ep_data_grp.create_dataset("actions", data=actions)

        # store model xml directly in the new hdf5 file
        model_path = os.path.join(ref_folder, "models", f["data/{}".format(ep)].attrs["model_file"])
        f_model = open(model_path, "r")
        model_xml = f_model.read()
        f_model.close()
        ep_data_grp.attrs["model_file"] = model_xml

        # store num samples for this ep
        num_samples = actions.shape[0]
        ep_data_grp.attrs["num_samples"] = num_samples # number of transitions in this episode
        num_samples_arr.append(num_samples)

    # write dataset attributes (metadata)
    f_new_grp.attrs["total"] = np.sum(num_samples_arr)

    # construct and save env metadata
    env_meta = dict()
    env_meta["type"] = EB.EnvType.ROBOSUITE_TYPE
    env_meta["env_name"] = (f["data"].attrs["env"] + "Teleop")
    # hardcode robosuite v0.3 args
    robosuite_args = {
        "has_renderer": False,
        "has_offscreen_renderer": False,
        "ignore_done": True,
        "use_object_obs": True,
        "use_camera_obs": False,
        "camera_depth": False,
        "camera_height": 84,
        "camera_width": 84,
        "camera_name": "agentview",
        "gripper_visualization": False,
        "reward_shaping": False,
        "control_freq": 100,
    }
    env_meta["env_kwargs"] = robosuite_args
    f_new_grp.attrs["env_args"] = json.dumps(env_meta, indent=4) # environment info

    print("\n====== Added env meta ======")
    print(f_new_grp.attrs["env_args"])

    f.close()
    f_new.close()

    # back up the old dataset, and replace with new dataset
    os.rename(hdf5_path, os.path.join(ref_folder, "demo_bak.hdf5"))
    os.rename(new_path, hdf5_path)


def split_fastest_from_hdf5(hdf5_path, n):
    """
    Creates filter key for fastest N trajectories, named
    "fastest_{}".format(n).

    Args:
        hdf5_path (str): path to the hdf5 file

        n (int): fastest n demos to create filter key for
    """

    # retrieve fastest n demos
    f = h5py.File(hdf5_path, "r")
    demos = sorted(list(f["data"].keys()))
    traj_lengths = []
    for ep in demos:
        traj_lengths.append(f["data/{}/actions".format(ep)].shape[0])
    inds = np.argsort(traj_lengths)[:n]
    filtered_demos = [demos[i] for i in inds]
    f.close()

    # create filter key
    name = "fastest_{}".format(n)
    lengths = create_hdf5_filter_key(hdf5_path=hdf5_path, demo_keys=filtered_demos, key_name=name)

    print("Total number of samples in fastest {} demos: {}".format(n, np.sum(lengths)))
    print("Average number of samples in fastest {} demos: {}".format(n, np.mean(lengths)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder",
        type=str,
        help="path to a folder containing a demo.hdf5 and a models directory containing \
            mujoco xml files. For example, RoboTurkPilot/bins-Can.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=225,
        help="creates a filter key corresponding to the n fastest trajectories. Defaults to 225.",
    )
    args = parser.parse_args()

    # convert hdf5
    convert_rt_pilot_hdf5(ref_folder=args.folder)

    # create 90-10 train-validation split in the dataset
    print("\nCreating 90-10 train-validation split...\n")
    hdf5_path = os.path.join(args.folder, "demo.hdf5")
    split_train_val_from_hdf5(hdf5_path=hdf5_path, val_ratio=0.1)

    print("\nCreating filter key for fastest {} trajectories...".format(args.n))
    split_fastest_from_hdf5(hdf5_path=hdf5_path, n=args.n)

    print("\nCreating 90-10 train-validation split for fastest {} trajectories...".format(args.n))
    split_train_val_from_hdf5(hdf5_path=hdf5_path, val_ratio=0.1, filter_key="fastest_{}".format(args.n))

    print(
        "\nWARNING: new dataset has replaced old one in demo.hdf5 file. "
        "The old dataset file has been moved to demo_bak.hdf5"
    )

    print(
        "\nNOTE: the new dataset also contains a fastest_{} filter key, for an easy way "
        "to train on the fastest trajectories. Just set config.train.hdf5_filter to train on this "
        "subset. A common choice is 225 when training on the bins-Can dataset.\n".format(args.n)
    )
