"""
Helper script to prepare datasets for diffusion policy training by (1) adding absolute actions and (2) 
writing the absolute actions to action dictionaries.
"""
import os
import h5py
import argparse
import numpy as np

import robomimic
from robomimic.scripts.conversion.extract_action_dict import extract_action_dict

import mimicgen
from mimicgen.scripts.add_datagen_info import add_datagen_info

DATASETS = [
    "/tmp/coffee/src_10.hdf5",
    "/tmp/stack/src_10.hdf5",
]


def convert_actions_in_dataset(dataset_path):
    """
    Helper function to call the relevant scripts to get absolute action dicts for a given dataset.
    """

    # first get absolute actions
    args = argparse.Namespace()
    args.dataset = dataset_path
    args.n = None
    args.absolute = True
    args.output = None
    add_datagen_info(args)

    # next convert actions to dict
    args = argparse.Namespace()
    args.dataset = dataset_path
    extract_action_dict(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        type=str,
        nargs='+',
        default=None,
    )
    args = parser.parse_args()

    datasets = args.datasets
    if datasets is None:
        datasets = DATASETS

    for d in datasets:
        dataset_path = os.path.expanduser(d)
        convert_actions_in_dataset(dataset_path)
