"""
Internal conversion tool to convert teleoperation config to env meta in demonstration hdf5s.
Also adds trajectory length to each episode as metadata if it is missing.
"""
import os
import h5py
import json
import argparse

import robosuite

import robomimic
import robomimic.envs.env_base as EB
from robomimic.config import Config
from robomimic.envs.env_robosuite import EnvRobosuite


def env_args_from_teleop_config(teleop_config, env_metadata=None):
    """
    Extract some important robosuite args from the teleop config.
    """

    # some default arguments that will be overriden when extracting data
    robosuite_args = {
        "has_renderer": False,
        "has_offscreen_renderer": False,
        "ignore_done": True,
        "use_object_obs": True,
        "use_camera_obs": False,
        "camera_depth": False,
        "camera_height": 84,
        "camera_width": 84,
        "camera_name": "agentview",
        "gripper_visualization": False,
        "reward_shaping": False,
    }

    is_v1 = (robosuite.__version__.split(".")[0] == "1")
    is_v1_1 = is_v1 and (robosuite.__version__.split(".")[1] == "1")
    is_v1_2 = is_v1 and (robosuite.__version__.split(".")[1] == "2")
    if env_metadata is not None and is_v1:
        # set some args
        robosuite_args["control_freq"] = env_metadata["robosuite_args"]["control_freq"]
        robosuite_args["controller_configs"] = env_metadata["robosuite_args"]["controller_configs"]
        robosuite_args["robots"] = env_metadata["robosuite_args"]["robots"]

        if "env_configuration" in env_metadata["robosuite_args"]:
            robosuite_args["env_configuration"] = env_metadata["robosuite_args"]["env_configuration"]

        # renaming of args
        if not is_v1_1 and not is_v1_2:
            robosuite_args["gripper_visualizations"] = robosuite_args["gripper_visualization"]
        robosuite_args["camera_names"] = robosuite_args["camera_name"]
        robosuite_args["camera_depths"] = robosuite_args["camera_depth"]
        robosuite_args["camera_heights"] = robosuite_args["camera_height"]
        robosuite_args["camera_widths"] = robosuite_args["camera_width"]
        del robosuite_args["gripper_visualization"]
        del robosuite_args["camera_name"]
        del robosuite_args["camera_depth"]
        del robosuite_args["camera_height"]
        del robosuite_args["camera_width"]

        return robosuite_args

    robosuite_args["control_freq"] = 100
    if teleop_config.controller.flag.osc:
        # using OSC - pass controller config to env args
        import RobotTeleop
        controller_json_path = os.path.join(RobotTeleop.__path__[0], 
                "assets/osc/robosuite/osc.json")
        with open(controller_json_path, "r") as f:
            controller_args = json.load(f)
        robosuite_args["controller_config"] = controller_args
        robosuite_args["control_freq"] = 20
    return robosuite_args


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        help="path to input hdf5 dataset",
    )
    args = parser.parse_args()

    # open file for editing to add env meta to metadata
    f = h5py.File(args.dataset, "a")

    # read env name
    env_name = f["data"].attrs["env"]

    # teleop config to robosuite args
    teleop_config_json = json.loads(f["data"].attrs["teleop_config"])
    teleop_config = Config(teleop_config_json)
    teleop_env_config_json = None
    if "teleop_env_metadata" in f["data"].attrs:
        teleop_env_config_json = json.loads(f["data"].attrs["teleop_env_metadata"])
    robosuite_args = env_args_from_teleop_config(teleop_config, env_metadata=teleop_env_config_json)

    # construct and save env meta
    env_meta = dict()
    env_meta["type"] = EB.EnvType.ROBOSUITE_TYPE
    env_meta["env_name"] = env_name
    env_meta["env_kwargs"] = robosuite_args
    f["data"].attrs["env_args"] = json.dumps(env_meta, indent=4) # environment info

    print("====== Added env meta ======")
    print(f["data"].attrs["env_args"])

    total_samples = 0
    for ep in f["data"]:
        # ensure model-xml is in per-episode metadata
        assert "model_file" in f["data/{}".format(ep)].attrs

        # add "num_samples" into per-episode metadata
        if "num_samples" in f["data/{}".format(ep)].attrs:
            del f["data/{}".format(ep)].attrs["num_samples"]
        n_sample = f["data/{}/actions".format(ep)].shape[0]
        f["data/{}".format(ep)].attrs["num_samples"] = n_sample
        total_samples += n_sample

    # add total samples to global metadata
    if "total" in f["data"].attrs:
        del f["data"].attrs["total"]
    f["data"].attrs["total"] = total_samples

    f.close()
