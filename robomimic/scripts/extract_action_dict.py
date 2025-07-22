"""
This script extracts action dictionaries from a dataset in HDF5 format.
It reads the actions from the dataset, processes them to extract position,
rotation (both as axis-angle and 6D representation), and gripper state,
and saves these as new datasets within the HDF5 file under the "action_dict"
group for each demonstration.
"""

import argparse
import pathlib
import sys
import tqdm
import h5py
import numpy as np
import torch
import os

import robomimic.utils.torch_utils as TorchUtils

def extract_action_dict(dataset, add_absolute_actions=True):
    f = h5py.File(os.path.expanduser(dataset), mode="r+")

    SPECS = [
        dict(
            key="actions",
            is_absolute=False,
        )
    ]
    if add_absolute_actions:
        SPECS.append(
            dict(
                key="actions_abs",
                is_absolute=True,
            )
        )

    # execute
    for spec in SPECS:
        input_action_key = spec["key"]
        is_absolute = spec["is_absolute"]

        if is_absolute:
            prefix = "abs_"
        else:
            prefix = "rel_"

        for demo in f["data"].values():
            in_action = demo[str(input_action_key)][:]
            in_pos = in_action[:,:3].astype(np.float32)
            in_rot = in_action[:,3:6].astype(np.float32)
            in_grip = in_action[:,6:7].astype(np.float32)
            
            rot_6d = TorchUtils.axis_angle_to_rot_6d(
                axis_angle=torch.from_numpy(in_rot)
            )
            rot_6d = rot_6d.numpy().astype(np.float32) # convert to numpy
            
            this_action_dict = {
                prefix + "pos": in_pos,
                prefix + "rot_axis_angle": in_rot,
                prefix + "rot_6d": rot_6d,
                "gripper": in_grip
            }

            # special case: 8 dim actions mean there is a mobile base mode in the action space
            if in_action.shape[1] == 8:
                this_action_dict["base_mode"] = in_action[:,7:8].astype(np.float32)

            action_dict_group = demo.require_group("action_dict")
            for key, data in this_action_dict.items():
                if key in action_dict_group:
                    del action_dict_group[key]
                action_dict_group.create_dataset(key, data=data)

    f.close()


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--dataset",
        type=str,
        required=True
    )
    
    args = parser.parse_args()
    extract_action_dict(args.dataset)
    
if __name__ == "__main__":
    main()
