"""
Script for replacing part of env_meta.
"""

import argparse
import h5py
import json


def main(hdf5_path):
    """
    Replace part of env-meta.
    """

    # list of all demonstration episodes (sorted in increasing number order)
    f = h5py.File(hdf5_path, "a")

    env_meta = json.loads(f["data"].attrs["env_args"])
    env_meta["env_name"] = "ToolHang"
    f["data"].attrs["env_args"] = json.dumps(env_meta, indent=4)

    f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        help="path to hdf5 dataset",
    )
    args = parser.parse_args()
    main(args.dataset)
