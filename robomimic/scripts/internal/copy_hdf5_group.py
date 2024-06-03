"""
Copy a group from one hdf5 to another
"""
import os
import argparse
import h5py
import numpy as np

def copy_hdf5_group(args):
    f_src = h5py.File(args.src)
    f_target = h5py.File(args.target, "a")
    
    for ep in f_src["data"].keys():
        if args.group not in f_target["data"][ep]:
            f_target["data"][ep].create_group(args.group)
        ep_group = f_target["data"][ep][args.group]
        for k in f_src["data"][ep][args.group].keys():
            if k not in ep_group:
                f_src["data"][ep][args.group].copy(k, ep_group, name=k)

    f_src.close()
    f_target.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # directory to download datasets to
    parser.add_argument(
        "--src",
        type=str,
        help="source hdf5",
    )

    parser.add_argument(
        "--target",
        type=str,
        help="target hdf5",
    )

    parser.add_argument(
        "--group",
        type=str,
        help="group to copy",
    )

    args = parser.parse_args()
    copy_hdf5_group(args)