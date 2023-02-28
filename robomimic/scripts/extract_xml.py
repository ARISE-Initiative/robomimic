import argparse
import os
import json
import h5py
import argparse
import imageio
import numpy as np

import robomimic
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
from robomimic.envs.env_base import EnvBase, EnvType


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        help="path to hdf5 dataset",
    )
    parser.add_argument(
        "--ignore_ds",
        action='store_true',
        help="ignore xml model in dataset",
    )
    # parser.add_argument(
    #     "--output_path",
    #     type=str,
    #     help="path to output xml file",
    # )
    args = parser.parse_args()

    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=args.dataset)
    env_type = EnvUtils.get_env_type(env_meta=env_meta)

    # need to make sure ObsUtils knows which observations are images, but it doesn't matter 
    # for playback since observations are unused. Pass a dummy spec here.
    dummy_spec = dict(
        obs=dict(
                low_dim=["robot0_eef_pos"],
                rgb=[],
            ),
    )
    ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs=dummy_spec)

    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=args.dataset)
    env = EnvUtils.create_env_from_metadata(env_meta=env_meta, render=False, render_offscreen=True)

    # some operations for playback are robosuite-specific, so determine if this environment is a robosuite env
    is_robosuite_env = EnvUtils.is_robosuite_env(env_meta)

    # use the wrapped env

    if args.ignore_ds:
        env.reset()
    else:
        f = h5py.File(args.dataset, "r")
        ep = "demo_0"
        states = f["data/{}/states".format(ep)][()]
        initial_state = dict(states=states[0])
        initial_state["model"] = f["data/{}".format(ep)].attrs["model_file"]
        env.reset()
        env.reset_to(initial_state)

    xml = env.env.sim.model.get_xml()
    print(xml)

    env.env.close()

