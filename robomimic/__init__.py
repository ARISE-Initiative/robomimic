__version__ = "0.1.0"


# stores released dataset links and rollout horizons in global dictionary. Structure is given below:
# {
#   task:
#       dataset_type:
#           hdf5_type:
#               url: link
#               horizon: value
#           ...
#       ...
#   ...
# }
DATASET_REGISTRY = {}


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
    ph_tasks = ["lift", "can", "square", "transport", "tool_hang", "lift_real", "can_real", "tool_hang_real"]
    ph_horizons = [400, 400, 400, 700, 700, 1000, 1000, 1000]
    for task, horizon in zip(ph_tasks, ph_horizons):
        register_dataset_link(task=task, dataset_type="ph", hdf5_type="raw", horizon=horizon,
            link="http://downloads.cs.stanford.edu/downloads/rt_benchmark/{}/ph/demo.hdf5".format(task))
        # real world datasets only have demo.hdf5 files which already contain all observation modalities
        # while sim datasets store raw low-dim mujoco states in the demo.hdf5
        if "real" not in task:
            register_dataset_link(task=task, dataset_type="ph", hdf5_type="low_dim", horizon=horizon,
                link="http://downloads.cs.stanford.edu/downloads/rt_benchmark/{}/ph/low_dim.hdf5".format(task))
            register_dataset_link(task=task, dataset_type="ph", hdf5_type="image", horizon=horizon,
                link="http://downloads.cs.stanford.edu/downloads/rt_benchmark/{}/ph/image.hdf5".format(task))

    # all multi human datasets
    mh_tasks = ["lift", "can", "square", "transport"]
    mh_horizons = [500, 500, 500, 1100]
    for task, horizon in zip(mh_tasks, mh_horizons):
        register_dataset_link(task=task, dataset_type="mh", hdf5_type="raw", horizon=horizon,
            link="http://downloads.cs.stanford.edu/downloads/rt_benchmark/{}/mh/demo.hdf5".format(task))
        register_dataset_link(task=task, dataset_type="mh", hdf5_type="low_dim", horizon=horizon,
            link="http://downloads.cs.stanford.edu/downloads/rt_benchmark/{}/mh/low_dim.hdf5".format(task))
        register_dataset_link(task=task, dataset_type="mh", hdf5_type="image", horizon=horizon,
            link="http://downloads.cs.stanford.edu/downloads/rt_benchmark/{}/mh/image.hdf5".format(task))

    # all machine generated datasets
    for task, horizon in zip(["lift", "can"], [400, 400]):
        register_dataset_link(task=task, dataset_type="mg", hdf5_type="raw", horizon=horizon,
            link="http://downloads.cs.stanford.edu/downloads/rt_benchmark/{}/mg/demo.hdf5".format(task))
        register_dataset_link(task=task, dataset_type="mg", hdf5_type="low_dim_sparse", horizon=horizon,
            link="http://downloads.cs.stanford.edu/downloads/rt_benchmark/{}/mg/low_dim_sparse.hdf5".format(task))
        register_dataset_link(task=task, dataset_type="mg", hdf5_type="image_sparse", horizon=horizon,
            link="http://downloads.cs.stanford.edu/downloads/rt_benchmark/{}/mg/image_sparse.hdf5".format(task))
        register_dataset_link(task=task, dataset_type="mg", hdf5_type="low_dim_dense", horizon=horizon,
            link="http://downloads.cs.stanford.edu/downloads/rt_benchmark/{}/mg/low_dim_dense.hdf5".format(task))
        register_dataset_link(task=task, dataset_type="mg", hdf5_type="image_dense", horizon=horizon,
            link="http://downloads.cs.stanford.edu/downloads/rt_benchmark/{}/mg/image_dense.hdf5".format(task))

    # can-paired dataset
    register_dataset_link(task="can", dataset_type="paired", hdf5_type="raw", horizon=400,
        link="http://downloads.cs.stanford.edu/downloads/rt_benchmark/can/paired/demo.hdf5")
    register_dataset_link(task="can", dataset_type="paired", hdf5_type="low_dim", horizon=400,
        link="http://downloads.cs.stanford.edu/downloads/rt_benchmark/can/paired/low_dim.hdf5")
    register_dataset_link(task="can", dataset_type="paired", hdf5_type="image", horizon=400,
        link="http://downloads.cs.stanford.edu/downloads/rt_benchmark/can/paired/image.hdf5")


register_all_links()