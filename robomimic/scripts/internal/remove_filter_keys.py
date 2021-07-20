"""
A convenience script to remove filter keys from a file.
"""

import h5py
import argparse


def main(args):

    f = h5py.File(args.dataset, "a")
    for fk in args.filter_keys:
        assert "mask/{}".format(fk) in f, "filter key {} not found!".format(fk)
        if args.valid:
            assert "mask/{}_train".format(fk) in f, "filter key {}_train not found!".format(fk)
            assert "mask/{}_valid".format(fk) in f, "filter key {}_valid not found!".format(fk)
            del f["mask/{}_train".format(fk)]
            del f["mask/{}_valid".format(fk)]
        del f["mask/{}".format(fk)]
    f.close()


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
        help="filter keys to remove",
    )
    parser.add_argument(
        "--valid",
        action='store_true',
        help="remove train-valid splits as well",
    )

    args = parser.parse_args()
    main(args)

