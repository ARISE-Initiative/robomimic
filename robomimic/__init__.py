__version__ = "0.5.0"


# stores released dataset links and rollout horizons in global dictionary.
# Structure is given below for each type of dataset:

# robosuite / real
# {
#   task:
#       dataset_type:
#           hdf5_type:
#               url: path in Hugging Face repo
#               horizon: value
#           ...
#       ...
#   ...
# }
DATASET_REGISTRY = {}

# momart
# {
#   task:
#       dataset_type:
#           url: link
#           size: value
#       ...
#   ...
# }
MOMART_DATASET_REGISTRY = {}

# Hugging Face repo ID
HF_REPO_ID = "amandlek/robomimic"


def register_dataset_link(task, dataset_type, hdf5_type, link, horizon):
    """
    Helper function to register dataset link in global dictionary.
    Also takes a @horizon parameter - this corresponds to the evaluation
    rollout horizon that should be used during training.

    Args:
        task (str): name of task for this dataset
        dataset_type (str): type of dataset (usually identifies the dataset source)
        hdf5_type (str): type of hdf5 - usually one of "raw", "low_dim", or "image",
            to identify the kind of observations in the dataset
        link (str): download link for the dataset
        horizon (int): evaluation rollout horizon that should be used with this dataset
    """
    if task not in DATASET_REGISTRY:
        DATASET_REGISTRY[task] = {}
    if dataset_type not in DATASET_REGISTRY[task]:
        DATASET_REGISTRY[task][dataset_type] = {}
    DATASET_REGISTRY[task][dataset_type][hdf5_type] = dict(url=link, horizon=horizon)


def register_all_links():
    """
    Record all dataset links in this function.
    """

    # all proficient human datasets
    ph_tasks = ["lift", "can", "square", "transport", "tool_hang"]
    ph_horizons = [400, 400, 400, 700, 700]
    for task, horizon in zip(ph_tasks, ph_horizons):
        register_dataset_link(task=task, dataset_type="ph", hdf5_type="raw", horizon=horizon,
            link="v1.5/{}/ph/demo_v15.hdf5".format(
                task,
            )
        )
        register_dataset_link(task=task, dataset_type="ph", hdf5_type="low_dim", horizon=horizon,
            link="v1.5/{}/ph/low_dim_v15.hdf5".format(task))
        register_dataset_link(task=task, dataset_type="ph", hdf5_type="image", horizon=horizon,
            link=None)

    ph_real_tasks = ["lift_real", "can_real", "tool_hang_real"]
    ph_real_horizons = [1000, 1000, 1000]
    for task, horizon in zip(ph_real_tasks, ph_real_horizons):
        # note: real-world datasets are hosted on stanford server, not HF
        register_dataset_link(task=task, dataset_type="ph", hdf5_type="raw", horizon=horizon,
            link="http://downloads.cs.stanford.edu/downloads/rt_benchmark/{}/ph/demo.hdf5".format(
                task,
            )
        )

    # all multi human datasets
    mh_tasks = ["lift", "can", "square", "transport"]
    mh_horizons = [500, 500, 500, 1100]
    for task, horizon in zip(mh_tasks, mh_horizons):
        register_dataset_link(task=task, dataset_type="mh", hdf5_type="raw", horizon=horizon,
            link="v1.5/{}/mh/demo_v15.hdf5".format(task))
        register_dataset_link(task=task, dataset_type="mh", hdf5_type="low_dim", horizon=horizon,
            link="v1.5/{}/mh/low_dim_v15.hdf5".format(task))
        register_dataset_link(task=task, dataset_type="mh", hdf5_type="image", horizon=horizon,
            link=None)

    # all machine generated datasets
    for task, horizon in zip(["lift", "can"], [400, 400]):
        register_dataset_link(task=task, dataset_type="mg", hdf5_type="raw", horizon=horizon,
            link="v1.5/{}/mg/demo_v15.hdf5".format(task))
        register_dataset_link(task=task, dataset_type="mg", hdf5_type="low_dim_sparse", horizon=horizon,
            link="v1.5/{}/mg/low_dim_sparse_v15.hdf5".format(task))
        register_dataset_link(task=task, dataset_type="mg", hdf5_type="image_sparse", horizon=horizon,
            link=None)
        register_dataset_link(task=task, dataset_type="mg", hdf5_type="low_dim_dense", horizon=horizon,
            link="v1.5/{}/mg/low_dim_dense_v15.hdf5".format(task))
        register_dataset_link(task=task, dataset_type="mg", hdf5_type="image_dense", horizon=horizon,
            link=None)

    # can-paired dataset
    register_dataset_link(task="can", dataset_type="paired", hdf5_type="raw", horizon=400,
        link="v1.5/can/paired/demo_v15.hdf5")
    register_dataset_link(task="can", dataset_type="paired", hdf5_type="low_dim", horizon=400,
        link="v1.5/can/paired/low_dim_v15.hdf5")
    register_dataset_link(task="can", dataset_type="paired", hdf5_type="image", horizon=400,
        link=None)


def register_momart_dataset_link(task, dataset_type, link, dataset_size):
    """
    Helper function to register dataset link in global dictionary.
    Also takes a @horizon parameter - this corresponds to the evaluation
    rollout horizon that should be used during training.

    Args:
        task (str): name of task for this dataset
        dataset_type (str): type of dataset (usually identifies the dataset source)
        link (str): download link for the dataset
        dataset_size (float): size of the dataset, in GB
    """
    if task not in MOMART_DATASET_REGISTRY:
        MOMART_DATASET_REGISTRY[task] = {}
    if dataset_type not in MOMART_DATASET_REGISTRY[task]:
        MOMART_DATASET_REGISTRY[task][dataset_type] = {}
    MOMART_DATASET_REGISTRY[task][dataset_type] = dict(url=link, size=dataset_size)


def register_all_momart_links():
    """
    Record all dataset links in this function.
    """
    # all tasks, mapped to their [exp, sub, gen, sam] sizes
    momart_tasks = {
        "table_setup_from_dishwasher": [14, 14, 3.3, 0.6],
        "table_setup_from_dresser": [16, 17, 3.1, 0.7],
        "table_cleanup_to_dishwasher": [23, 36, 5.3, 1.1],
        "table_cleanup_to_sink": [17, 28, 2.9, 0.8],
        "unload_dishwasher": [21, 27, 5.4, 1.0],
    }

    momart_dataset_types = [
        "expert",
        "suboptimal",
        "generalize",
        "sample",
    ]

    # Iterate over all combos and register the link
    for task, dataset_sizes in momart_tasks.items():
        for dataset_type, dataset_size in zip(momart_dataset_types, dataset_sizes):
            register_momart_dataset_link(
                task=task,
                dataset_type=dataset_type,
                link=f"http://downloads.cs.stanford.edu/downloads/rt_mm/{dataset_type}/{task}_{dataset_type}.hdf5",
                dataset_size=dataset_size,
            )


register_all_links()
register_all_momart_links()
