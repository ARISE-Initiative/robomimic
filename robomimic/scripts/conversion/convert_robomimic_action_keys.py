import argparse
import pathlib
import sys
import tqdm
import h5py
import numpy as np
import torch
import pytorch3d.transforms as pt

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--glob",
        type=str,
        required=True
    )
    
    parser.add_argument(
        "--in_action_key",
        type=str,
        default='actions'
    )
    
    parser.add_argument(
        '--is_absolute',
        default=False
    )
    
    args = parser.parse_args()
    
    # find files
    file_paths = list(pathlib.Path.cwd().glob(args.glob))
    
    # confirm with the user
    print("Found matching files:")
    for f in file_paths:
        print(f)
        
    prefix = 'rel_'
    if args.is_absolute:
        prefix = 'abs_'
        
    print(f"Are you sure to modify these files with action prefix {prefix} ?")
    result = input("[y/n]?")
    if 'y' not in result:
        sys.exit(0)
    
    # execute
    for file_path in tqdm.tqdm(file_paths):
        with h5py.File(str(file_path), mode='r+') as file:
            for demo_key, demo in file['data'].items():
                in_action = demo[str(args.in_action_key)][:]
                in_pos = in_action[:,:3].astype(np.float32)
                in_rot = in_action[:,3:6].astype(np.float32)
                in_grip = in_action[:,6:].astype(np.float32)
                
                rot_ = torch.from_numpy(in_rot)
                rot_mat = pt.axis_angle_to_matrix(rot_)
                rot_6d = pt.matrix_to_rotation_6d(rot_mat).numpy().astype(np.float32)
                
                this_action_dict = {
                    prefix + 'pos': in_pos,
                    prefix + 'rot_axis_angle': in_rot,
                    prefix + 'rot_6d': rot_6d,
                    'gripper': in_grip
                }
                if 'action_dict' in demo:
                    del demo['action_dict']
                action_dict_group = demo.require_group('action_dict')
                for key, data in this_action_dict.items():
                    if key in action_dict_group:
                        del action_dict_group[key]
                    action_dict_group.create_dataset(key, data=data)
    
if __name__ == "__main__":
    main()
