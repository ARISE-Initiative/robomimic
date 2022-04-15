"""
Helper script to convert a dataset collected using robosuite into an hdf5 compatible with
this repository. Takes a dataset path corresponding to the demo.hdf5 file containing the
demonstrations. It modifies the dataset in-place. By default, the script also creates a
90-10 train-validation split.

For more information on collecting datasets with robosuite, see the code link and documentation
link below.

Code: https://github.com/ARISE-Initiative/robosuite/blob/offline_study/robosuite/scripts/collect_human_demonstrations.py

Documentation: https://robosuite.ai/docs/algorithms/demonstrations.html

Example usage:

    python convert_robosuite.py --dataset /path/to/your/demo.hdf5
"""

import h5py
import json
import argparse

import robomimic.envs.env_base as EB
from robomimic.scripts.split_train_val import split_train_val_from_hdf5


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        help="path to input hdf5 dataset",
    )
    args = parser.parse_args()

    f = h5py.File(args.dataset, "a") # edit mode

    # store env meta
    env_name = f["data"].attrs["env"]
    env_info = json.loads(f["data"].attrs["env_info"])
    env_meta = dict(
        type=EB.EnvType.ROBOSUITE_TYPE,
        env_name=env_name,
        env_kwargs=env_info,
    )
    if "env_args" in f["data"].attrs:
        del f["data"].attrs["env_args"]
    f["data"].attrs["env_args"] = json.dumps(env_meta, indent=4)

    print("====== Stored env meta ======")
    print(f["data"].attrs["env_args"])

    # store metadata about number of samples
    total_samples = 0
    for ep in f["data"]:
        # ensure model-xml is in per-episode metadata
        assert "model_file" in f["data/{}".format(ep)].attrs

        # add "num_samples" into per-episode metadata
        if "num_samples" in f["data/{}".format(ep)].attrs:
            del f["data/{}".format(ep)].attrs["num_samples"]
        n_sample = f["data/{}/actions".format(ep)].shape[0]
        f["data/{}".format(ep)].attrs["num_samples"] = n_sample
        total_samples += n_sample

    # add total samples to global metadata
    if "total" in f["data"].attrs:
        del f["data"].attrs["total"]
    f["data"].attrs["total"] = total_samples

    f.close()

    # create 90-10 train-validation split in the dataset
    split_train_val_from_hdf5(hdf5_path=args.dataset, val_ratio=0.1)
