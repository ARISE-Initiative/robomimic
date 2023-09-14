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
from robomimic.scripts.conversion.robosuite_add_absolute_actions import add_absolute_actions_to_dataset
from robomimic.scripts.conversion.extract_action_dict import extract_action_dict
from robomimic.scripts.filter_dataset_size import filter_dataset_size


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        help="path to input hdf5 dataset",
    )
    parser.add_argument(
        "--filter_num_demos",
        type=int,
        nargs='+',
        help="Num demos to filter by (can be list)",
    )
    args = parser.parse_args()

    f = h5py.File(args.dataset, "a") # edit mode

    # store env meta
    env_name = f["data"].attrs.get("env", None)
    if "env_info" in f["data"].attrs:
        env_info = json.loads(f["data"].attrs["env_info"])
    if env_name is not None and env_info is not None:
        env_meta = dict(
            type=EB.EnvType.ROBOSUITE_TYPE,
            env_name=env_name,
            env_version=f["data"].attrs["repository_version"],
            env_kwargs=env_info,
        )
        if "env_args" in f["data"].attrs:
            del f["data"].attrs["env_args"]
        f["data"].attrs["env_args"] = json.dumps(env_meta, indent=4)
    else:
        # assume env_args already present
        assert "env_args" in f["data"].attrs

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

    # add absolute actions to dataset
    add_absolute_actions_to_dataset(
        dataset=args.dataset,
        eval_dir=None,
        num_workers=10,
    )

    # extract corresponding action keys into action_dict
    extract_action_dict(dataset=args.dataset)

    # create filter keys according to number of demos
    if args.filter_num_demos is not None:
        for n in args.filter_num_demos:
            filter_dataset_size(
                args.dataset,
                num_demos=n,
            )