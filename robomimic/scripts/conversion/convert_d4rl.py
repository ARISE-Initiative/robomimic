"""
Helper script to convert D4RL data into an hdf5 compatible with this repository.
Takes a folder path and a D4RL env name. This script downloads the corresponding
raw D4RL dataset into a "d4rl" subfolder, and then makes a converted dataset 
in the "d4rl/converted" subfolder.

This script has been tested on the follwing commits:

    https://github.com/rail-berkeley/d4rl/tree/9b68f31bab6a8546edfb28ff0bd9d5916c62fd1f
    https://github.com/rail-berkeley/d4rl/tree/26adf732efafdad864b3df2287e7b778ee4f7f63

Args:
    env (str): d4rl env name, which specifies the dataset to download and convert
    folder (str): specify folder to download raw d4rl datasets and converted d4rl datasets to.
        A `d4rl` subfolder will be created in this folder with the raw d4rl dataset, and 
        a `d4rl/converted` subfolder will be created in this folder with the converted
        datasets (if they do not already exist). Defaults to the datasets folder at
        the top-level of the repository.

Example usage:

    # downloads to default path at robomimic/datasets/d4rl
    python convert_d4rl.py --env walker2d-medium-expert-v0

    # download to custom path
    python convert_d4rl.py --env walker2d-medium-expert-v0 --folder /path/to/folder
"""

import os
import h5py
import json
import argparse
import numpy as np

import gym
import d4rl
import robomimic
from robomimic.envs.env_gym import EnvGym
from robomimic.utils.log_utils import custom_tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env",
        type=str,
        help="d4rl env name, which specifies the dataset to download and convert",
    )
    parser.add_argument(
        "--folder",
        type=str,
        default=None,
        help="specify folder to download raw d4rl datasets and converted d4rl datasets to.\
            A `d4rl` subfolder will be created in this folder with the raw d4rl dataset, and\
            a `d4rl/converted` subfolder will be created in this folder with the converted\
            datasets (if they do not already exist). Defaults to the datasets folder at\
            the top-level of the repository.",
    )
    args = parser.parse_args()

    base_folder = args.folder
    if base_folder is None:
        base_folder = os.path.join(robomimic.__path__[0], "../datasets")
    base_folder = os.path.join(base_folder, "d4rl")

    # get dataset
    d4rl.set_dataset_path(base_folder)
    env = gym.make(args.env)
    ds = env.env.get_dataset()
    env.close()

    # env
    env = EnvGym(args.env)

    # output file
    write_folder = os.path.join(base_folder, "converted")
    if not os.path.exists(write_folder):
        os.makedirs(write_folder)
    output_path = os.path.join(base_folder, "converted", "{}.hdf5".format(args.env.replace("-", "_")))
    f_sars = h5py.File(output_path, "w")
    f_sars_grp = f_sars.create_group("data")

    # code to split D4RL data into trajectories
    # (modified from https://github.com/aviralkumar2907/d4rl_evaluations/blob/bear_intergrate/bear/examples/bear_hdf5_d4rl.py#L18)
    all_obs = ds['observations']
    all_act = ds['actions']
    N = all_obs.shape[0]

    obs = all_obs[:N-1]
    actions = all_act[:N-1]
    next_obs = all_obs[1:]
    rewards = np.squeeze(ds['rewards'][:N-1])
    dones = np.squeeze(ds['terminals'][:N-1]).astype(np.int32)

    assert 'timeouts' in ds
    timeouts = ds['timeouts'][:]

    ctr = 0
    total_samples = 0
    num_traj = 0
    traj = dict(obs=[], next_obs=[], actions=[], rewards=[], dones=[])

    print("\nConverting hdf5...")
    for idx in custom_tqdm(range(obs.shape[0])):

        # add transition
        traj["obs"].append(obs[idx])
        traj["actions"].append(actions[idx])
        traj["rewards"].append(rewards[idx])
        traj["next_obs"].append(next_obs[idx])
        traj["dones"].append(dones[idx])
        ctr += 1

        # if hit timeout or done is True, end the current trajectory and start a new trajectory
        if timeouts[idx] or dones[idx]:

            # replace next obs with copy of current obs for final timestep, and make sure done is true
            traj["next_obs"][-1] = np.array(obs[idx])
            traj["dones"][-1] = 1

            # store trajectory
            ep_data_grp = f_sars_grp.create_group("demo_{}".format(num_traj))
            ep_data_grp.create_dataset("obs/flat", data=np.array(traj["obs"]))
            ep_data_grp.create_dataset("next_obs/flat", data=np.array(traj["next_obs"]))
            ep_data_grp.create_dataset("actions", data=np.array(traj["actions"]))
            ep_data_grp.create_dataset("rewards", data=np.array(traj["rewards"]))
            ep_data_grp.create_dataset("dones", data=np.array(traj["dones"]))
            ep_data_grp.attrs["num_samples"] = len(traj["actions"])
            total_samples += len(traj["actions"])
            num_traj += 1

            # reset
            ctr = 0
            traj = dict(obs=[], next_obs=[], actions=[], rewards=[], dones=[])

    print("\nExcluding {} samples at end of file due to no trajectory truncation.".format(len(traj["actions"])))
    print("Wrote {} trajectories to new converted hdf5 at {}\n".format(num_traj, output_path))

    # metadata
    f_sars_grp.attrs["total"] = total_samples
    f_sars_grp.attrs["env_args"] = json.dumps(env.serialize(), indent=4)

    f_sars.close()

