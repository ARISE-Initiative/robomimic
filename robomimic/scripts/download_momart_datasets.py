"""
Script to download datasets used in MoMaRT paper (https://arxiv.org/abs/2112.05251). By default, all
datasets will be stored at robomimic/datasets, unless the @download_dir
argument is supplied. We recommend using the default, as most examples that
use these datasets assume that they can be found there.

The @tasks and @dataset_types arguments can all be supplied
to choose which datasets to download. 

Args:
    download_dir (str): Base download directory. Created if it doesn't exist. 
        Defaults to datasets folder in repository - only pass in if you would
        like to override the location.

    tasks (list): Tasks to download datasets for. Defaults to table_setup_from_dishwasher task. Pass 'all' to
        download all tasks - 5 total:
            - table_setup_from_dishwasher
            - table_setup_from_dresser
            - table_cleanup_to_dishwasher
            - table_cleanup_to_sink
            - unload_dishwasher
    
    dataset_types (list): Dataset types to download datasets for (expert, suboptimal, generalize, sample).
        Defaults to expert. Pass 'all' to download datasets for all available dataset
        types per task, or directly specify the list of dataset types.
        NOTE: Because these datasets are huge, we will always print out a warning
        that a user must respond yes to to acknowledge the data size (can be up to >100G for all tasks of a single type)

Example usage:

    # default behavior - just download expert table_setup_from_dishwasher dataset
    python download_momart_datasets.py

    # download expert datasets for all tasks
    # (do a dry run first to see which datasets would be downloaded)
    python download_momart_datasets.py --tasks all --dataset_types expert --dry_run
    python download_momart_datasets.py --tasks all --dataset_types expert low_dim

    # download all expert and suboptimal datasets for the table_setup_from_dishwasher and table_cleanup_to_dishwasher tasks
    python download_datasets.py --tasks table_setup_from_dishwasher table_cleanup_to_dishwasher --dataset_types expert suboptimal

    # download the sample datasets
    python download_datasets.py --tasks all --dataset_types sample

    # download all datasets
    python download_datasets.py --tasks all --dataset_types all
"""
import os
import argparse

import robomimic
import robomimic.utils.file_utils as FileUtils
from robomimic import MOMART_DATASET_REGISTRY

ALL_TASKS = [
    "table_setup_from_dishwasher",
    "table_setup_from_dresser",
    "table_cleanup_to_dishwasher",
    "table_cleanup_to_sink",
    "unload_dishwasher",
]
ALL_DATASET_TYPES = [
    "expert",
    "suboptimal",
    "generalize",
    "sample",
]


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
        default=["table_setup_from_dishwasher"],
        help="Tasks to download datasets for. Defaults to table_setup_from_dishwasher task. Pass 'all' to download all"
             f"5 tasks, or directly specify the list of tasks. Options are any of: {ALL_TASKS}",
    )

    # dataset types to download datasets for
    parser.add_argument(
        "--dataset_types",
        type=str,
        nargs='+',
        default=["expert"],
        help="Dataset types to download datasets for (e.g. expert, suboptimal). Defaults to expert. Pass 'all' to "
             "download datasets for all available dataset types per task, or directly specify the list of dataset "
             f"types. Options are any of: {ALL_DATASET_TYPES}",
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

    download_dataset_types = args.dataset_types
    if "all" in download_dataset_types:
        assert len(download_dataset_types) == 1, "all should be only dataset_types argument but got: {}".format(args.dataset_types)
        download_dataset_types = ALL_DATASET_TYPES

    # Run sanity check first to warn user if they're about to download a huge amount of data
    total_size = 0
    for task in MOMART_DATASET_REGISTRY:
        if task in download_tasks:
            for dataset_type in MOMART_DATASET_REGISTRY[task]:
                if dataset_type in download_dataset_types:
                    total_size += MOMART_DATASET_REGISTRY[task][dataset_type]["size"]

    # Verify user acknowledgement if we're not doing a dry run
    if not args.dry_run:
        user_response = input(f"Warning: requested datasets will take a total of {total_size}GB. Proceed? y/n\n")
        assert user_response.lower() in {"yes", "y"}, f"Did not receive confirmation. Aborting download."

    # download requested datasets
    for task in MOMART_DATASET_REGISTRY:
        if task in download_tasks:
            for dataset_type in MOMART_DATASET_REGISTRY[task]:
                if dataset_type in download_dataset_types:
                    dataset_info = MOMART_DATASET_REGISTRY[task][dataset_type]
                    download_dir = os.path.abspath(os.path.join(default_base_dir, task, dataset_type))
                    print(f"\nDownloading dataset:\n"
                          f"    task: {task}\n"
                          f"    dataset type: {dataset_type}\n"
                          f"    dataset size: {dataset_info['size']}GB\n"
                          f"    download path: {download_dir}")
                    if args.dry_run:
                        print("\ndry run: skip download")
                    else:
                        # Make sure path exists and create if it doesn't
                        os.makedirs(download_dir, exist_ok=True)
                        FileUtils.download_url(
                            url=dataset_info["url"],
                            download_dir=download_dir,
                        )
                    print("")
