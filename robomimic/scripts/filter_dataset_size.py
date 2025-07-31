"""
This script filters a dataset in HDF5 format to create subsets of a specified number of demonstrations.
It reads the dataset, randomly selects a specified number of demonstrations, and creates a new filter key
for the subset. The script can also take an input filter key to filter a specific subset of demonstrations
and can output the results with a custom filter key name.
"""

import argparse
import h5py
import numpy as np

from robomimic.utils.file_utils import create_hdf5_filter_key


def filter_dataset_size(hdf5_path, num_demos, input_filter_key=None, output_filter_key=None):
    # retrieve number of demos
    f = h5py.File(hdf5_path, "r")
    if input_filter_key is not None:
        print("using filter key: {}".format(input_filter_key))
        demos = sorted([elem.decode("utf-8") for elem in np.array(f["mask/{}".format(input_filter_key)])])
    else:
        demos = sorted(list(f["data"].keys()))
    f.close()

    # get random split
    total_num_demos = len(demos)
    mask = np.zeros(total_num_demos)
    mask[:num_demos] = 1.
    np.random.shuffle(mask)
    mask = mask.astype(int)
    subset_inds = mask.nonzero()[0]
    subset_keys = [demos[i] for i in subset_inds]

    # pass mask to generate split
    if output_filter_key is not None:
        name = output_filter_key
    else:
        name = "{}_demos".format(num_demos)

    if input_filter_key is not None:
        name = "{}_{}".format(input_filter_key, name)

    subset_lengths = create_hdf5_filter_key(hdf5_path=hdf5_path, demo_keys=subset_keys, key_name=name)

    print("Total number of subset samples: {}".format(np.sum(subset_lengths)))
    print("Average number of subset samples {}".format(np.mean(subset_lengths)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="path to hdf5 dataset",
    )
    parser.add_argument(
        "--input_filter_key",
        type=str,
        default=None,
        help="if provided, split the subset of trajectories in the file that correspond to\
            this filter key into a training and validation set of trajectories, instead of\
            splitting the full set of trajectories",
    )
    parser.add_argument(
        "--num_demos",
        type=int,
        nargs='+',
        required=True,
    )
    parser.add_argument(
        "--output_filter_key",
        type=str,
        required=False,
        help="(optional) use custom name for output filter key name"
    )
    args = parser.parse_args()

    # seed to make sure results are consistent
    np.random.seed(0)

    for n in args.num_demos:
        filter_dataset_size(
            args.dataset,
            input_filter_key=args.input_filter_key,
            num_demos=n,
            output_filter_key=args.output_filter_key,
        )