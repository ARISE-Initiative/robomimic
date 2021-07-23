"""
Script for getting an hdf5 filter key for fastest N demos.

Args:
    dataset (str): path to hdf5 dataset

    n (int): fastest n demos to create filter key for

Example usage:

    # creates filter key "fastest_225" for the 225 trajectories with lowest length
    python split_fastest.py --dataset /path/to/demo.hdf5 --n 225
"""

import argparse
import h5py
import numpy as np

from robomimic.utils.file_utils import create_hdf5_filter_key


def split_fastest_from_hdf5(hdf5_path, n):
    """
    Creates filter key for fastest N trajectories.

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
        "--dataset",
        type=str,
        help="path to hdf5 dataset",
    )
    parser.add_argument(
        "--n",
        type=int,
        help="number of fastest demos to create filter key for"
    )
    args = parser.parse_args()

    split_fastest_from_hdf5(hdf5_path=args.dataset, n=args.n)
