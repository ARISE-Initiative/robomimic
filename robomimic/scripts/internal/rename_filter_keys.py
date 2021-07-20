"""
A convenience script to rename existing filter keys.
"""

import h5py
import argparse
import numpy as np


def main(args):

    assert len(args.filter_keys) == len(args.rename_filter_keys), "mismatch in filter key argument length!"

    f = h5py.File(args.dataset, "a")
    for fk, rfk in zip(args.filter_keys, args.rename_filter_keys):
        # get fk demos
        assert "mask/{}".format(fk) in f, "filter key {} not found!".format(fk)
        fk_demos = sorted([elem.decode("utf-8") for elem in np.array(f["mask/{}".format(fk)])])

        if args.valid:
            # get fk train-val demos
            assert "mask/{}_train".format(fk) in f, "filter key {}_train not found!".format(fk)
            assert "mask/{}_valid".format(fk) in f, "filter key {}_valid not found!".format(fk)
            train_fk_demos = sorted([x.decode("utf-8") for x in f["mask/{}_train".format(fk)][:]])
            valid_fk_demos = sorted([x.decode("utf-8") for x in f["mask/{}_valid".format(fk)][:]])
            assert set(train_fk_demos + valid_fk_demos) == set(fk_demos), "train-val split has issues"

            # replace train-val fk
            f["mask/{}_train".format(rfk)] = np.array(train_fk_demos, dtype='S')
            f["mask/{}_valid".format(rfk)] = np.array(valid_fk_demos, dtype='S')
            del f["mask/{}_train".format(fk)]
            del f["mask/{}_valid".format(fk)]

        # replace fk
        f["mask/{}".format(rfk)] = np.array(fk_demos, dtype='S')
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
        help="filter keys to rename",
    )
    parser.add_argument(
        "--rename_filter_keys",
        type=str,
        nargs='+',
        help="new filter key names",
    )
    parser.add_argument(
        "--valid",
        action='store_true',
        help="rename train-valid splits as well",
    )

    args = parser.parse_args()
    main(args)

