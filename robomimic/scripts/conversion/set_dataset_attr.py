"""
Example:
python robomimic/scripts/set_dataset_attr.py --glob 'datasets/**/*_abs.hdf5' --env_args env_kwargs.controller_configs.control_delta=false absolute_actions=true 
"""
import argparse
import pathlib
import json
import sys
import tqdm
import h5py

def update_env_args_dict(env_args_dict: dict, key: tuple, value):
    if key is None:
        return env_args_dict
    elif len(key) == 0:
        return env_args_dict
    elif len(key) == 1:
        env_args_dict[key[0]] = value
        return env_args_dict
    else:
        this_key = key[0]
        if this_key not in env_args_dict:
            env_args_dict[this_key] = dict()
        update_env_args_dict(env_args_dict[this_key], key[1:], value)
        return env_args_dict

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--glob",
        type=str,
        required=True
    )
    
    parser.add_argument(
        "--env_args",
        type=str,
        default=None
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
        
    # parse env_args update
    env_args_key = None
    env_args_value = None
    if args.env_args is not None:
        key, svalue = args.env_args.split('=')
        env_args_key = key.split('.')
        env_args_value = json.loads(svalue)
    
    # find files
    file_paths = list(pathlib.Path.cwd().glob(args.glob))
    
    # confirm with the user
    print("Found matching files:")
    for f in file_paths:
        print(f)
    print("Are you sure to modify these files with the following attributes:")
    print(json.dumps(attrs_dict, indent=2))
    if env_args_key is not None:
        print("env_args."+'.'.join(env_args_key)+'='+str(env_args_value))
    result = input("[y/n]?")
    if 'y' not in result:
        sys.exit(0)
    
    # execute
    for file_path in tqdm.tqdm(file_paths):
        with h5py.File(str(file_path), mode='r+') as file:
            # update env_args
            if env_args_key is not None:
                env_args = file['data'].attrs['env_args']
                env_args_dict = json.loads(env_args)
                env_args_dict = update_env_args_dict(
                    env_args_dict=env_args_dict, 
                    key=env_args_key, value=env_args_value)
                env_args = json.dumps(env_args_dict)
                file['data'].attrs['env_args'] = env_args
            
            # update other attrs
            file['data'].attrs.update(attrs_dict)
    
if __name__ == "__main__":
    main()
