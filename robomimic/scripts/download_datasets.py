"""
Script to download datasets packaged with the repository. By default, all
datasets will be stored at robomimic/datasets, unless the @download_dir
argument is supplied. We recommend using the default, as most examples that
use these datasets assume that they can be found there.

The @tasks, @dataset_types, and @hdf5_types arguments can all be supplied
to choose which datasets to download. 

Args:
    download_dir (str): Base download directory. Created if it doesn't exist. 
        Defaults to datasets folder in repository - only pass in if you would
        like to override the location.

    tasks (list): Tasks to download datasets for. Defaults to lift task. Pass 'all' to 
        download all tasks (sim + real) 'sim' to download all sim tasks, 'real' to 
        download all real tasks, or directly specify the list of tasks.
    
    dataset_types (list): Dataset types to download datasets for (e.g. ph, mh, mg). 
        Defaults to ph. Pass 'all' to download datasets for all available dataset 
        types per task, or directly specify the list of dataset types.

    hdf5_types (list): hdf5 types to download datasets for (e.g. raw, low_dim, image). 
        Defaults to low_dim. Pass 'all' to download datasets for all available hdf5 
        types per task and dataset, or directly specify the list of hdf5 types.

Example usage:

    # default behavior - just download lift proficient-human low-dim dataset
    python download_datasets.py

    # download low-dim proficient-human datasets for all simulation tasks
    # (do a dry run first to see which datasets would be downloaded)
    python download_datasets.py --tasks sim --dataset_types ph --hdf5_types low_dim --dry_run
    python download_datasets.py --tasks sim --dataset_types ph --hdf5_types low_dim

    # download all low-dim and image multi-human datasets for the can and square tasks
    python download_datasets.py --tasks can square --dataset_types mh --hdf5_types low_dim image

    # download the sparse reward machine-generated low-dim datasets
    python download_datasets.py --tasks all --dataset_types mg --hdf5_types low_dim_sparse

    # download all real robot datasets
    python download_datasets.py --tasks real
"""
import os
import argparse

import robomimic
import robomimic.utils.file_utils as FileUtils
from robomimic import DATASET_REGISTRY

ALL_TASKS = ["lift", "can", "square", "transport", "tool_hang", "lift_real", "can_real", "tool_hang_real"]
ALL_DATASET_TYPES = ["ph", "mh", "mg", "paired"]
ALL_HDF5_TYPES = ["raw", "low_dim", "image", "low_dim_sparse", "low_dim_dense", "image_sparse", "image_dense"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # directory to download datasets to
    parser.add_argument(
        "--download_dir",
        type=str,
        default=None,
        help="Base download directory. Created if it doesn't exist. Defaults to datasets folder in repository.",
    )

    # tasks to download datasets for
    parser.add_argument(
        "--tasks",
        type=str,
        nargs='+',
        default=["lift"],
        help="Tasks to download datasets for. Defaults to lift task. Pass 'all' to download all tasks (sim + real)\
            'sim' to download all sim tasks, 'real' to download all real tasks, or directly specify the list of\
            tasks.",
    )

    # dataset types to download datasets for
    parser.add_argument(
        "--dataset_types",
        type=str,
        nargs='+',
        default=["ph"],
        help="Dataset types to download datasets for (e.g. ph, mh, mg). Defaults to ph. Pass 'all' to download \
            datasets for all available dataset types per task, or directly specify the list of dataset types.",
    )

    # hdf5 types to download datasets for
    parser.add_argument(
        "--hdf5_types",
        type=str,
        nargs='+',
        default=["low_dim"],
        help="hdf5 types to download datasets for (e.g. raw, low_dim, image). Defaults to raw. Pass 'all' \
            to download datasets for all available hdf5 types per task and dataset, or directly specify the list\
            of hdf5 types.",
    )

    # dry run - don't actually download datasets, but print which datasets would be downloaded
    parser.add_argument(
        "--dry_run",
        action='store_true',
        help="set this flag to do a dry run to only print which datasets would be downloaded"
    )

    args = parser.parse_args()

    # set default base directory for downloads
    default_base_dir = args.download_dir
    if default_base_dir is None:
        default_base_dir = os.path.join(robomimic.__path__[0], "../datasets")

    # load args
    download_tasks = args.tasks
    if "all" in download_tasks:
        assert len(download_tasks) == 1, "all should be only tasks argument but got: {}".format(args.tasks)
        download_tasks = ALL_TASKS
    elif "sim" in download_tasks:
        assert len(download_tasks) == 1, "sim should be only tasks argument but got: {}".format(args.tasks)
        download_tasks = [task for task in ALL_TASKS if "real" not in task]
    elif "real" in download_tasks:
        assert len(download_tasks) == 1, "real should be only tasks argument but got: {}".format(args.tasks)
        download_tasks = [task for task in ALL_TASKS if "real" in task]

    download_dataset_types = args.dataset_types
    if "all" in download_dataset_types:
        assert len(download_dataset_types) == 1, "all should be only dataset_types argument but got: {}".format(args.dataset_types)
        download_dataset_types = ALL_DATASET_TYPES

    download_hdf5_types = args.hdf5_types
    if "all" in download_hdf5_types:
        assert len(download_hdf5_types) == 1, "all should be only hdf5_types argument but got: {}".format(args.hdf5_types)
        download_hdf5_types = ALL_HDF5_TYPES

    # download requested datasets
    for task in DATASET_REGISTRY:
        if task in download_tasks:
            for dataset_type in DATASET_REGISTRY[task]:
                if dataset_type in download_dataset_types:
                    for hdf5_type in DATASET_REGISTRY[task][dataset_type]:
                        if hdf5_type in download_hdf5_types:
                            download_dir = os.path.abspath(os.path.join(default_base_dir, task, dataset_type))
                            print("\nDownloading dataset:\n    task: {}\n    dataset type: {}\n    hdf5 type: {}\n    download path: {}"
                                .format(task, dataset_type, hdf5_type, download_dir))
                            url = DATASET_REGISTRY[task][dataset_type][hdf5_type]["url"]
                            if url is None:
                                print(
                                    "Skipping {}-{}-{}, no url for dataset exists.".format(task, dataset_type, hdf5_type)
                                    + " Create this dataset locally by running the appropriate command from robomimic/scripts/extract_obs_from_raw_datasets.sh."
                                )
                                continue
                            if args.dry_run:
                                print("\ndry run: skip download")
                            else:
                                # Make sure path exists and create if it doesn't
                                os.makedirs(download_dir, exist_ok=True)
                                FileUtils.download_url(
                                    url=DATASET_REGISTRY[task][dataset_type][hdf5_type]["url"], 
                                    download_dir=download_dir,
                                )
                            print("")
