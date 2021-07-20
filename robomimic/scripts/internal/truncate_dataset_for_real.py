"""
Script for truncating an hdf5 into first N trajectories (not filter key) and removing observations
from hdf5. Make sure to call h5repack to actually save disk space!
"""

import argparse
import h5py
import numpy as np


def truncate_hdf5(hdf5_path, n):
    # list of all demonstration episodes (sorted in increasing number order)
    f = h5py.File(hdf5_path, "a")
    demos = list(f["data"].keys())
    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]

    # assert number of demos is more than n
    num_demos = len(demos)
    assert num_demos > n, "less demos ({}) than truncation number ({})!".format(num_demos, n)

    print("truncating {} demos to first {}".format(num_demos, n))

    for ep in demos[n:]:
        del f["data/{}".format(ep)]
    f.close()

    print("WARNING: make sure to call h5repack to actually reclaim the space!")


def remove_obs(hdf5_path, obs_keys):
    """
    Removes obs_keys (list) as observations from the hdf5. Useful for
    real robot hdf5s that collect extraneous, unneeded observations.
    """

    # list of all demonstration episodes (sorted in increasing number order)
    f = h5py.File(hdf5_path, "a")
    demos = list(f["data"].keys())
    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]

    print("removing {} from observations".format(obs_keys))

    for ep in demos:
        for k in obs_keys:
            del f["data/{}/obs/{}".format(ep, k)]
            del f["data/{}/next_obs/{}".format(ep, k)]
    f.close()

    print("WARNING: make sure to call h5repack to actually reclaim the space!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        help="path to hdf5 dataset",
    )
    parser.add_argument(
        "--n",
        type=int,
        help="number of demos to truncate to",
        default=None,
    )
    parser.add_argument(
        "--obs_keys",
        type=str,
        help="observations to remove from the file",
        nargs='+',
        default=[],
    )
    args = parser.parse_args()

    if args.n is not None:
        truncate_hdf5(args.dataset, n=args.n)
    if len(args.obs_keys) > 0:
        remove_obs(args.dataset, obs_keys=args.obs_keys)