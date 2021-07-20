"""
Script for replacing part of env_meta.
"""

import argparse
import h5py
import json


def main(hdf5_path, verbose=False):
    """
    Prints attributes present in hdf5 file.
    """

    # list of all demonstration episodes (sorted in increasing number order)
    f = h5py.File(hdf5_path, "r")

    global_attr_keys = [k for k in f["data"].attrs]
    print("\nglobal attributes")
    print(global_attr_keys)

    if verbose:
        for k in global_attr_keys:
            print("")
            print("key: {}".format(k))
            print(f["data"].attrs[k])

    ep = [k for k in f["data"]][0]
    demo_attr_keys = [k for k in f["data/{}".format(ep)].attrs]
    print("\ndemo attributes")
    print(demo_attr_keys)

    if verbose:
        for k in demo_attr_keys:
            print("")
            print("key: {}".format(k))
            print(f["data/{}".format(ep)].attrs[k])

    f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        help="path to hdf5 dataset",
    )
    parser.add_argument(
        "--verbose",
        action='store_true',
        help="verbose output",
    )
    args = parser.parse_args()
    main(args.dataset, args.verbose)
