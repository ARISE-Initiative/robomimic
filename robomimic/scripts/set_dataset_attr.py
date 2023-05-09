"""
Example:
python robomimic/scripts/set_dataset_attr.py --glob 'datasets/**/*_abs.hdf5' absolute_actions=true
"""

import argparse
import pathlib
import json
import sys
import tqdm
import h5py

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--glob",
        type=str,
        required=True
    )
    
    parser.add_argument(
        'attrs',
        nargs='*'
    )
    
    args = parser.parse_args()
    
    # parse attrs to set
    # format: key=value
    # values are parsed with json
    attrs_dict = dict()
    for attr_arg in args.attrs:
        key, svalue = attr_arg.split("=")
        value = json.loads(svalue)
        attrs_dict[key] = value
    
    # find files
    file_paths = list(pathlib.Path.cwd().glob(args.glob))
    
    # confirm with the user
    print("Found matching files:")
    for f in file_paths:
        print(f)
    print("Are you sure to modify these files with the following attributes:")
    print(json.dumps(attrs_dict, indent=2))
    result = input("[y/n]?")
    if 'y' not in result:
        sys.exit(0)
    
    # execute
    for file_path in tqdm.tqdm(file_paths):
        with h5py.File(str(file_path), mode='r+') as file:
            file['data'].attrs.update(attrs_dict)
            # print(file['data'].attrs['absolute_actions'])
    
if __name__ == "__main__":
    main()
