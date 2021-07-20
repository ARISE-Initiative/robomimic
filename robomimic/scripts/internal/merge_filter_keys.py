"""
A convenience script to create a new filter key in a file by merging two or more existing filter keys.
"""

import h5py
import argparse
import numpy as np

from robomimic.utils.file_utils import create_hdf5_filter_key


def main(args):

    # aggregate the demonstration keys across the filter keys
    f = h5py.File(args.dataset, "r")
    all_demos = []
    all_train_demos = []
    all_valid_demos = []
    for fk in args.filter_keys:
        assert "mask/{}".format(fk) in f, "filter key {} not found!".format(fk)
        demos = sorted([elem.decode("utf-8") for elem in np.array(f["mask/{}".format(fk)])])
        all_demos += demos
        if args.valid:
            assert "mask/{}_train".format(fk) in f, "filter key {}_train not found!".format(fk)
            assert "mask/{}_valid".format(fk) in f, "filter key {}_valid not found!".format(fk)
            train_demos = [x.decode("utf-8") for x in f["mask/{}_train".format(fk)][:]]
            valid_demos = [x.decode("utf-8") for x in f["mask/{}_valid".format(fk)][:]]
            assert set(train_demos + valid_demos) == set(demos), "train-val split has issues"
            all_train_demos += train_demos
            all_valid_demos += valid_demos
    f.close()

    # make sure we have no redundant keys
    assert len(set(all_demos)) == len(all_demos), "redundant keys present among filter keys!"

    # write new filter keys
    print("creating new filter key {} from filter keys {}".format(args.name, args.filter_keys))
    create_hdf5_filter_key(hdf5_path=args.dataset, demo_keys=sorted(all_demos), key_name=args.name)
    if args.valid:
        print("creating new filter keys {} and {}".format("{}_train".format(args.name), "{}_valid".format(args.name)))
        create_hdf5_filter_key(hdf5_path=args.dataset, demo_keys=sorted(all_train_demos), key_name="{}_train".format(args.name))
        create_hdf5_filter_key(hdf5_path=args.dataset, demo_keys=sorted(all_valid_demos), key_name="{}_valid".format(args.name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        help="path to hdf5 dataset",
    )
    parser.add_argument(
        "--filter_keys",
        type=str,
        nargs='+',
        help="filter keys to merge into a new filter key",
    )
    parser.add_argument(
        "--name",
        type=str,
        help="name for filter key to create",
    )
    parser.add_argument(
        "--valid",
        action='store_true',
        help="merge train-valid splits as well",
    )

    args = parser.parse_args()
    main(args)

