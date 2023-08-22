import argparse
import pathlib
import sys
import tqdm
import h5py
import numpy as np
import torch
import os

def extract_action_dict(args):    
    # find files
    f = h5py.File(os.path.expanduser(args.dataset), mode="r+")

    SPECS = [
        dict(
            key="actions",
            is_absolute=False,
        ),
        dict(
            key="actions_abs",
            is_absolute=True,
        )
    ]

    # execute
    for spec in SPECS:
        input_action_key = spec["key"]
        is_absolute = spec["is_absolute"]

        if is_absolute:
            prefix = "abs_"
        else:
            prefix = "rel_"

        for demo in f['data'].values():
            in_action = demo[str(input_action_key)][:]
            in_pos = in_action[:,:3].astype(np.float32)
            in_rot = in_action[:,3:6].astype(np.float32)
            in_grip = in_action[:,6:].astype(np.float32)

            rot_ = torch.from_numpy(in_rot)
            rot_6d = TorchUtils.axis_angle_to_rot_6d(rot_).numpy().astype(np.float32)
            
            this_action_dict = {
                prefix + 'pos': in_pos,
                prefix + 'rot_axis_angle': in_rot,
                prefix + 'rot_6d': rot_6d,
                'gripper': in_grip
            }
            # if 'action_dict' in demo:
            #     del demo['action_dict']
            action_dict_group = demo.require_group('action_dict')
            for key, data in this_action_dict.items():
                if key in action_dict_group:
                    del action_dict_group[key]
                action_dict_group.create_dataset(key, data=data)

    f.close()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--dataset",
        type=str,
        required=True
    )
    
    args = parser.parse_args()

    extract_action_dict(args)
