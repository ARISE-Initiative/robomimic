"""
Remove the MG_ env prefix from env args
"""

import json
import h5py
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        help="path to input hdf5 dataset",
    )
    args = parser.parse_args()

    f = h5py.File(args.dataset, "a") # edit mode

    env_args = f["data"].attrs["env_args"]
    env_args = json.loads(env_args)
    env_name = env_args["env_name"]
    if env_name.startswith("MG_"):
        env_args["env_name"] = env_name[3:]

    f["data"].attrs["env_args"] = json.dumps(env_args)

    f.close()