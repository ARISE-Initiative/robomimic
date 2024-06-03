import h5py
import numpy as np
import argparse
import json
from collections import OrderedDict

from robomimic.utils.file_utils import create_hdf5_filter_key

SPEC = OrderedDict(
    PnPCounterToCab=dict(
        exclude_obj_groups=["condiment_bottle", "baguette", "kettle_electric", "avocado", "can"],
    ),
    PnPCabToCounter=dict(
        exclude_obj_groups=["beer", "orange", "jam", "canned_food", "coffee_cup"],
    ),
    PnPCounterToSink=dict(
        exclude_obj_groups=["apple", "banana", "bar_soap", "cup", "cucumber"],
    ),
    PnPSinkToCounter=dict(
        exclude_obj_groups=["peach", "lime", "yogurt", "fish", "kiwi"],
    ),
    PnPCounterToMicrowave=dict(
        exclude_obj_groups=["broccoli", "cheese", "bell_pepper", "squash", "sweet_potato"],
    ),
    PnPMicrowaveToCounter=dict(
        exclude_obj_groups=["corn", "tomato", "hot_dog", "egg", "carrot"],
    ),
    PnPCounterToStove=dict(
        exclude_obj_groups=["potato", "garlic", "steak", "eggplant", "mango"],
    ),
    PnPStoveToCounter=dict(
        exclude_obj_groups=["potato", "garlic", "steak", "eggplant", "mango"],
    ),
)

def add_train_filter_key(dataset):
    f = h5py.File(dataset)

    demos = sorted(list(f["data"].keys()))
    # put demonstration list in increasing episode order
    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]

    env_args = json.loads(f["data"].attrs["env_args"])
    env_name = env_args["env_name"]
    
    if env_name.startswith("MG_"):
        # remove the prefix for now
        env_name = env_name[3:]
    
    env_spec = SPEC.get(env_name, {})
    exclude_obj_groups = env_spec.get("exclude_obj_groups", [])
    train_demos = []
    
    for ep in demos:
        ep_meta = json.loads(f["data/{}".format(ep)].attrs["ep_meta"])
        obj_cfgs = ep_meta["object_cfgs"]
        obj_cat = None
        for cfg in obj_cfgs:
            if cfg["name"] == "obj":
                obj_cat = cfg["info"]["cat"]
                break
        layout_id = ep_meta["layout_id"]
        style_id = ep_meta["style_id"]
        
        if obj_cat in exclude_obj_groups:
            continue
    
        train_demos.append(ep)

    print("Total train demos:", len(train_demos))

    f.close()

    create_hdf5_filter_key(hdf5_path=dataset, demo_keys=train_demos, key_name="train")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="path to hdf5 dataset",
    )

    args = parser.parse_args()

    # seed to make sure results are consistent
    np.random.seed(0)

    add_train_filter_key(
        args.dataset,
    )