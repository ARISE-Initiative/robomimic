"""
Version of hyperparam helper to easily spin up runs with different base configs and diffusion policy.
"""
import os
import shutil
import json
import argparse

import robomimic
import robomimic.utils.hyperparam_utils as HyperparamUtils

import maglev_utils
from maglev_utils.utils.file_utils import config_generator_to_script_lines


# set base folder for where to copy each base config and generate new configs
CONFIG_DIR = "/tmp/diffusion_configs"

# path to base robomimic training config(s)
BASE_CONFIGS = [
    # "~/Desktop/mimicgen_env_data/base_train_diffusion.json",
    # "~/Desktop/mimicgen_env_data/base_train_diffusion_image.json",
    "~/Desktop/mimicgen_env_data/base_train_diffusion.json",
]

# output directory for this set of runs
OUTPUT_DIR = "/tmp/diffusion_runs"


def make_generators(base_configs):
    """Helper function to make all generators."""
    all_settings = [
        # # low-dim
        # dict(
        #     dataset_paths=[
        #         "/tmp/low_dim.hdf5",
        #     ],
        #     dataset_names=[
        #         "low_dim",
        #     ],
        #     horizon=400,
        # ),
        # # image
        # dict(
        #     dataset_paths=[
        #         "/tmp/image.hdf5",
        #     ],
        #     dataset_names=[
        #         "image",
        #     ],
        #     horizon=400,
        # ),
        dict(
            dataset_paths=[
                "/ext2/rebuttal/diffusion/square_ph_abs_im.hdf5",
            ],
            dataset_names=[
                "square_ph_ld",
            ],
            horizon=400,
        ),
    ]

    assert len(base_configs) == len(all_settings)
    ret = []
    for conf, setting in zip(base_configs, all_settings):
        ret.append(make_gen(os.path.expanduser(conf), setting))
    return ret


def make_gen(base_config, settings):
    """
    Specify training configs to generate here.
    """
    generator = HyperparamUtils.ConfigGenerator(
        base_config_file=base_config,
        script_file="", # will be overriden in next step
    )

    # add some params to sweep
    dataset_values = [[dict(path=x)] for x in settings["dataset_paths"]]
    generator.add_param(
        key="train.data", 
        name="ds", 
        group=0, 
        values=dataset_values,
        value_names=settings["dataset_names"],
    )

    # rollout settings
    generator.add_param(
        key="experiment.rollout.horizon", 
        name="", 
        group=1, 
        values=[settings["horizon"]],
    )

    # output path
    generator.add_param(
        key="train.output_dir",
        name="", 
        group=2, 
        values=[
            OUTPUT_DIR,
        ],
    )

    # ensure robosuite env uses absolute pose actions
    generator.add_param(
        key="experiment.env_meta_update_dict",
        name="",
        group=-1,
        values=[
            {"env_kwargs": {"controller_configs": {"control_delta": False}}}
        ],
    )

    # default action spec for diffusion policy
    generator.add_param(
        key="train.action_keys",
        name="",
        group=-1,
        values=[
            [
                "action_dict/abs_pos",
                "action_dict/abs_rot_6d",
                "action_dict/gripper",
                # "actions",
            ],
        ],
    )
    generator.add_param(
        key="train.action_config",
        name="",
        group=-1,
        values=[
            {
                "actions":{
                    "normalization": None,
                },
                "action_dict/abs_pos": {
                    "normalization": "min_max"
                },
                "action_dict/abs_rot_axis_angle": {
                    "normalization": "min_max",
                    "format": "rot_axis_angle"
                },
                "action_dict/abs_rot_6d": {
                    "normalization": None,
                    "format": "rot_6d"
                },
                "action_dict/rel_pos": {
                    "normalization": None,
                },
                "action_dict/rel_rot_axis_angle": {
                    "normalization": None,
                    "format": "rot_axis_angle"
                },
                "action_dict/rel_rot_6d": {
                    "normalization": None,
                    "format": "rot_6d"
                },
                "action_dict/gripper": {
                    "normalization": None,
                }
            }
        ],
    )

    # num data workers 4 by default (for both low-dim and image) and cache mode "low_dim"
    generator.add_param(
        key="train.num_data_workers",
        name="",
        group=-1,
        values=[4],
    )
    generator.add_param(
        key="train.hdf5_cache_mode",
        name="",
        group=-1,
        values=["low_dim"],
    )

    # num epochs 1000 for both low-dim and image
    generator.add_param(
        key="train.num_epochs",
        name="",
        group=-1,
        values=[1000],
    )

    # set low-rate of eval - every 100 epochs
    generator.add_param(
        key="experiment.save.every_n_epochs",
        name="",
        group=-1,
        values=[100],
    )
    generator.add_param(
        key="experiment.rollout.rate",
        name="",
        group=-1,
        values=[100],
    )

    # set noise scheduler
    use_ddim = True
    inf_steps = [(100, 10), (50, 5)]
    # use_ddim = False
    # inf_steps = []

    generator.add_param(
        key="algo.ddim.enabled",
        name="ddim" if use_ddim else "",
        group=1001,
        values=[
            use_ddim,
        ],
        value_names=[
            "t" if use_ddim else "f",
        ],
    )
    generator.add_param(
        key="algo.ddpm.enabled",
        name="ddpm" if not use_ddim else "",
        group=1001,
        values=[
            (not use_ddim),
        ],
        value_names=[
            "f" if not use_ddim else "t",
        ],
    )

    if len(inf_steps) > 0:
        train_inf_steps = [x[0] for x in inf_steps]
        eval_inf_steps = [x[1] for x in inf_steps]
        # set inf steps
        generator.add_param(
            key="algo.ddim.num_train_timesteps" if use_ddim else "algo.ddpm.num_train_timesteps",
            name="train",
            group=1002,
            values=train_inf_steps,
        )
        generator.add_param(
            key="algo.ddim.num_inference_timesteps" if use_ddim else "algo.ddpm.num_inference_timesteps",
            name="eval",
            group=1002,
            values=eval_inf_steps,
        )

    # # seed
    # generator.add_param(
    #     key="train.seed",
    #     name="seed", 
    #     group=100000, 
    #     values=[101, 102, 103],
    # )

    return generator


def main(args):

    # make config generators
    generators = make_generators(base_configs=BASE_CONFIGS)

    if args.config_dir is None:
        args.config_dir = CONFIG_DIR

    if os.path.exists(args.config_dir):
        ans = input("Non-empty dir at {} will be removed.\nContinue (y / n)? \n".format(args.config_dir))
        if ans != "y":
            exit()
        shutil.rmtree(args.config_dir)

    all_json_files, run_lines = config_generator_to_script_lines(generators, config_dir=args.config_dir)

    print("configs")
    print(json.dumps(all_json_files, indent=4))
    print("runs")
    print(json.dumps(run_lines, indent=4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Path to base json config - will override any defaults.
    parser.add_argument(
        "--config_dir",
        type=str,
        help="path to base config json that will be modified to generate jsons. The jsons will\
            be generated in the same folder as this file.",
    )

    args = parser.parse_args()
    main(args)
