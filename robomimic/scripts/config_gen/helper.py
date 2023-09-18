import argparse
import os
import time
import datetime

import robomimic
import robomimic.utils.hyperparam_utils as HyperparamUtils

base_path = os.path.abspath(os.path.join(os.path.dirname(robomimic.__file__), os.pardir))

def scan_datasets(folder, postfix=".h5"):
    dataset_paths = []
    for root, dirs, files in os.walk(os.path.expanduser(folder)):
        for f in files:
            if f.endswith(postfix):
                dataset_paths.append(os.path.join(root, f))
    return dataset_paths


def get_generator(algo_name, config_file, args, algo_name_short=None, pt=False):
    if args.wandb_proj_name is None:
        strings = [
            algo_name_short if (algo_name_short is not None) else algo_name,
            args.name,
            args.env,
            args.mod,
        ]
        args.wandb_proj_name = '_'.join([str(s) for s in strings if s is not None])

    if args.script is not None:
        generated_config_dir = os.path.join(os.path.dirname(args.script), "json")
    else:
        curr_time = datetime.datetime.fromtimestamp(time.time()).strftime('%m-%d-%y-%H-%M-%S')
        generated_config_dir=os.path.join(
            '~/', 'tmp/autogen_configs/ril', algo_name, args.env, args.mod, args.name, curr_time, "json",
        )

    generator = HyperparamUtils.ConfigGenerator(
        base_config_file=config_file,
        generated_config_dir=generated_config_dir,
        wandb_proj_name=args.wandb_proj_name,
        script_file=args.script,
    )

    args.algo_name = algo_name
    args.pt = pt

    return generator


def set_env_settings(generator, args):
    if args.env in ["r2d2"]:
        assert args.mod == "im"
        generator.add_param(
            key="experiment.rollout.enabled",
            name="",
            group=-1,
            values=[
                False
            ],
        )
        generator.add_param(
            key="experiment.save.every_n_epochs",
            name="",
            group=-1,
            values=[50],
        )
        generator.add_param(
            key="experiment.mse.enabled",
            name="",
            group=-1,
            values=[True],
        ),
        generator.add_param(
            key="experiment.mse.every_n_epochs",
            name="",
            group=-1,
            values=[50],
        ),
        generator.add_param(
            key="experiment.mse.on_save_ckpt",
            name="",
            group=-1,
            values=[True],
        ),
        generator.add_param(
            key="experiment.mse.num_samples",
            name="",
            group=-1,
            values=[20],
        ),
        generator.add_param(
            key="experiment.mse.visualize",
            name="",
            group=-1,
            values=[True],
        ),
        if "observation.modalities.obs.low_dim" not in generator.parameters:
            generator.add_param(
                key="observation.modalities.obs.low_dim",
                name="",
                group=-1,
                values=[
                    ["robot_state/cartesian_position", "robot_state/gripper_position"]
                ],
            )
        if "observation.modalities.obs.rgb" not in generator.parameters:
            generator.add_param(
                key="observation.modalities.obs.rgb",
                name="",
                group=-1,
                values=[
                    [
                        "camera/image/hand_camera_left_image",
                        "camera/image/varied_camera_1_left_image", "camera/image/varied_camera_2_left_image" # uncomment to use all 3 cameras
                    ]
                ],
            )
        generator.add_param(
            key="observation.encoder.rgb.obs_randomizer_class",
            name="obsrand",
            group=-1,
            values=[
                # "CropRandomizer", # crop only
                # "ColorRandomizer", # jitter only
                ["ColorRandomizer", "CropRandomizer"], # jitter, followed by crop
            ],
            hidename=True,
        )
        generator.add_param(
            key="observation.encoder.rgb.obs_randomizer_kwargs",
            name="obsrandargs",
            group=-1,
            values=[
                # {"crop_height": 116, "crop_width": 116, "num_crops": 1, "pos_enc": False}, # crop only
                # {}, # jitter only
                [{}, {"crop_height": 116, "crop_width": 116, "num_crops": 1, "pos_enc": False}], # jitter, followed by crop
            ],
            hidename=True,
        )
        if ("observation.encoder.rgb.obs_randomizer_kwargs" not in generator.parameters) and \
            ("observation.encoder.rgb.obs_randomizer_kwargs.crop_height" not in generator.parameters):
            generator.add_param(
                key="observation.encoder.rgb.obs_randomizer_kwargs.crop_height",
                name="",
                group=-1,
                values=[
                    116
                ],
            )
            generator.add_param(
                key="observation.encoder.rgb.obs_randomizer_kwargs.crop_width",
                name="",
                group=-1,
                values=[
                    116
                ],
            )
        # remove spatial softmax by default for r2d2 dataset
        generator.add_param(
            key="observation.encoder.rgb.core_kwargs.pool_class",
            name="",
            group=-1,
            values=[
                None
            ],
        )
        generator.add_param(
            key="observation.encoder.rgb.core_kwargs.pool_kwargs",
            name="",
            group=-1,
            values=[
                None
            ],
        )

        # specify dataset type is r2d2 rather than default robomimic
        generator.add_param(
            key="train.data_format",
            name="",
            group=-1,
            values=[
                "r2d2"
            ],
        )
        
        # here, we list how each action key should be treated (normalized etc)
        generator.add_param(
            key="train.action_config",
            name="",
            group=-1,
            values=[
                {
                    "action/cartesian_position":{
                        "normalization": "min_max",
                    },
                    "action/abs_pos":{
                        "normalization": "min_max",
                    },
                    "action/abs_rot_6d":{
                        "normalization": "min_max",
                        "format": "rot_6d",
                        "convert_at_runtime": "rot_euler",
                    },
                    "action/abs_rot_euler":{
                        "normalization": "min_max",
                        "format": "rot_euler",
                    },
                    "action/gripper_position":{
                        "normalization": "min_max",
                    },
                    "action/cartesian_velocity":{
                        "normalization": None,
                    },
                    "action/rel_pos":{
                        "normalization": None,
                    },
                    "action/rel_rot_6d":{
                        "format": "rot_6d",
                        "normalization": None,
                        "convert_at_runtime": "rot_euler",
                    },
                    "action/rel_rot_euler":{
                        "format": "rot_euler",
                        "normalization": None,
                    },
                    "action/gripper_velocity":{
                        "normalization": None,
                    },
                }
            ],
        )
        generator.add_param(
            key="train.dataset_keys",
            name="",
            group=-1,
            values=[[]],
        )
        if "train.action_keys" not in generator.parameters:
            generator.add_param(
                key="train.action_keys",
                name="ac_keys",
                group=-1,
                values=[
                    [
                        "action/rel_pos",
                        "action/rel_rot_euler",
                        "action/gripper_velocity",
                    ],
                ],
                value_names=[
                    "rel",
                ],
            )
        # observation key groups to swap
        generator.add_param(
            key="train.shuffled_obs_key_groups",
            name="",
            group=-1,
            values=[[[
                (
                "camera/image/varied_camera_1_left_image",
                "camera/image/varied_camera_1_right_image",
                "camera/extrinsics/varied_camera_1_left",
                "camera/extrinsics/varied_camera_1_right",
                ),
                (
                "camera/image/varied_camera_2_left_image",
                "camera/image/varied_camera_2_right_image",
                "camera/extrinsics/varied_camera_2_left",
                "camera/extrinsics/varied_camera_2_right",
                ),  
            ]]],
        )
    elif args.env == "kitchen":
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
                    },
                    "action_dict/base_mode": {
                        "normalization": None,
                    }
                }
            ],
        )
        
        if args.mod == 'im':
            generator.add_param(
                key="observation.modalities.obs.low_dim",
                name="",
                group=-1,
                values=[
                    ["robot0_eef_pos",
                     "robot0_eef_quat",
                     "robot0_base_pos",
                     "robot0_gripper_qpos"]
                ],
            )
            generator.add_param(
                key="observation.modalities.obs.rgb",
                name="",
                group=-1,
                values=[
                    ["robot0_agentview_left_image",
                     "robot0_agentview_right_image",
                     "robot0_eye_in_hand_image"]
                ],
            )
        else:
            generator.add_param(
                key="observation.modalities.obs.low_dim",
                name="",
                group=-1,
                values=[
                    ["robot0_eef_pos",
                     "robot0_eef_quat",
                     "robot0_gripper_qpos",
                     "robot0_base_pos",
                     "object",
                    ]
                ],
            )
    elif args.env in ['square', 'lift', 'place_close']:
        # # set videos off
        # args.no_video = True

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

        if args.mod == 'im':
            generator.add_param(
                key="observation.modalities.obs.low_dim",
                name="",
                group=-1,
                values=[
                    ["robot0_eef_pos",
                     "robot0_eef_quat",
                     "robot0_gripper_qpos"]
                ],
            )
            generator.add_param(
                key="observation.modalities.obs.rgb",
                name="",
                group=-1,
                values=[
                    ["agentview_image",
                     "robot0_eye_in_hand_image"]
                ],
            )
        else:
            generator.add_param(
                key="observation.modalities.obs.low_dim",
                name="",
                group=-1,
                values=[
                    ["robot0_eef_pos",
                     "robot0_eef_quat",
                     "robot0_gripper_qpos",
                     "object"]
                ],
            )
    elif args.env == 'transport':
        # set videos off
        args.no_video = True

        # TODO: fix 2 robot case
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

        if args.mod == 'im':
            generator.add_param(
                key="observation.modalities.obs.low_dim",
                name="",
                group=-1,
                values=[
                    ["robot0_eef_pos",
                     "robot0_eef_quat",
                     "robot0_gripper_qpos",
                     "robot1_eef_pos",
                     "robot1_eef_quat",
                     "robot1_gripper_qpos"]
                ],
            )
            generator.add_param(
                key="observation.modalities.obs.rgb",
                name="",
                group=-1,
                values=[
                    ["shouldercamera0_image",
                     "robot0_eye_in_hand_image",
                     "shouldercamera1_image",
                     "robot1_eye_in_hand_image"]
                ],
            )
        else:
            generator.add_param(
                key="observation.modalities.obs.low_dim",
                name="",
                group=-1,
                values=[
                    ["robot0_eef_pos",
                     "robot0_eef_quat",
                     "robot0_gripper_qpos",
                     "robot1_eef_pos",
                     "robot1_eef_quat",
                     "robot1_gripper_qpos",
                     "object"]
                ],
            )

        generator.add_param(
            key="experiment.rollout.horizon",
            name="",
            group=-1,
            values=[700],
        )
    elif args.env == 'tool_hang':
        # set videos off
        args.no_video = True

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

        if args.mod == 'im':
            generator.add_param(
                key="observation.modalities.obs.low_dim",
                name="",
                group=-1,
                values=[
                    ["robot0_eef_pos",
                     "robot0_eef_quat",
                     "robot0_gripper_qpos"]
                ],
            )
            generator.add_param(
                key="observation.modalities.obs.rgb",
                name="",
                group=-1,
                values=[
                    ["sideview_image",
                     "robot0_eye_in_hand_image"]
                ],
            )
            generator.add_param(
                key="observation.encoder.rgb.obs_randomizer_kwargs.crop_height",
                name="",
                group=-1,
                values=[
                    216
                ],
            )
            generator.add_param(
                key="observation.encoder.rgb.obs_randomizer_kwargs.crop_width",
                name="",
                group=-1,
                values=[
                    216
                ],
            )
            generator.add_param(
                key="observation.encoder.rgb2.obs_randomizer_kwargs.crop_height",
                name="",
                group=-1,
                values=[
                    216
                ],
            )
            generator.add_param(
                key="observation.encoder.rgb2.obs_randomizer_kwargs.crop_width",
                name="",
                group=-1,
                values=[
                    216
                ],
            )
        else:
            generator.add_param(
                key="observation.modalities.obs.low_dim",
                name="",
                group=-1,
                values=[
                    ["robot0_eef_pos",
                     "robot0_eef_quat",
                     "robot0_gripper_qpos",
                     "object"]
                ],
            )

        generator.add_param(
            key="experiment.rollout.horizon",
            name="",
            group=-1,
            values=[700],
        )
    else:
        raise ValueError


def set_mod_settings(generator, args):
    if args.mod == 'ld':
        if "experiment.save.epochs" not in generator.parameters:
            generator.add_param(
                key="experiment.save.epochs",
                name="",
                group=-1,
                values=[
                    [2000]
                ],
            )
    elif args.mod == 'im':
        if "experiment.save.every_n_epochs" not in generator.parameters:
            generator.add_param(
                key="experiment.save.every_n_epochs",
                name="",
                group=-1,
                values=[40],
            )

        generator.add_param(
            key="experiment.epoch_every_n_steps",
            name="",
            group=-1,
            values=[500],
        )
        if "train.num_data_workers" not in generator.parameters:
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
        if "train.batch_size" not in generator.parameters:
            generator.add_param(
                key="train.batch_size",
                name="",
                group=-1,
                values=[16],
            )
        if "train.num_epochs" not in generator.parameters:
            generator.add_param(
                key="train.num_epochs",
                name="",
                group=-1,
                values=[600],
            )
        if "experiment.rollout.rate" not in generator.parameters:
            generator.add_param(
                key="experiment.rollout.rate",
                name="",
                group=-1,
                values=[40],
            )


def set_debug_mode(generator, args):
    if not args.debug:
        return

    generator.add_param(
        key="experiment.mse.every_n_epochs",
        name="",
        group=-1,
        values=[2],
        value_names=[""],
    )
    generator.add_param(
        key="experiment.mse.visualize",
        name="",
        group=-1,
        values=[True],
        value_names=[""],
    )
    generator.add_param(
        key="experiment.rollout.n",
        name="",
        group=-1,
        values=[2],
        value_names=[""],
    )
    generator.add_param(
        key="experiment.rollout.horizon",
        name="",
        group=-1,
        values=[30],
        value_names=[""],
    )
    generator.add_param(
        key="experiment.rollout.rate",
        name="",
        group=-1,
        values=[2],
        value_names=[""],
    )
    generator.add_param(
        key="experiment.epoch_every_n_steps",
        name="",
        group=-1,
        values=[2],
        value_names=[""],
    )
    generator.add_param(
        key="experiment.save.every_n_epochs",
        name="",
        group=-1,
        values=[2],
        value_names=[""],
    )
    generator.add_param(
        key="experiment.validation_epoch_every_n_steps",
        name="",
        group=-1,
        values=[2],
        value_names=[""],
    )
    generator.add_param(
        key="train.num_epochs",
        name="",
        group=-1,
        values=[2],
        value_names=[""],
    )
    if args.name is None:
        generator.add_param(
            key="experiment.name",
            name="",
            group=-1,
            values=["debug"],
            value_names=[""],
        )
    generator.add_param(
        key="experiment.save.enabled",
        name="",
        group=-1,
        values=[False],
        value_names=[""],
    )
    generator.add_param(
        key="train.hdf5_cache_mode",
        name="",
        group=-1,
        values=["low_dim"],
        value_names=[""],
    )
    generator.add_param(
        key="train.num_data_workers",
        name="",
        group=-1,
        values=[3],
    )


def set_output_dir(generator, args):
    assert args.name is not None

    vals = generator.parameters["train.output_dir"].values

    for i in range(len(vals)):
        vals[i] = os.path.join(vals[i], args.name)


def set_wandb_mode(generator, args):
    generator.add_param(
        key="experiment.logging.log_wandb",
        name="",
        group=-1,
        values=[not args.no_wandb],
    )


def set_num_seeds(generator, args):
    if args.n_seeds is not None and "train.seed" not in generator.parameters:
        generator.add_param(
            key="train.seed",
            name="seed",
            group=-10,
            values=[i + 1 for i in range(args.n_seeds)],
            prepend=True,
        )


def get_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--name",
        type=str,
    )

    parser.add_argument(
        "--env",
        type=str,
        default='r2d2',
    )

    parser.add_argument(
        '--mod',
        type=str,
        choices=['ld', 'im'],
        default='im',
    )

    parser.add_argument(
        "--ckpt_mode",
        type=str,
        choices=["off", "all", "best_only"],
        default=None,
    )

    parser.add_argument(
        "--script",
        type=str,
        default=None
    )

    parser.add_argument(
        "--wandb_proj_name",
        type=str,
        default=None
    )

    parser.add_argument(
        "--debug",
        action="store_true",
    )

    parser.add_argument(
        '--no_video',
        action='store_true'
    )

    parser.add_argument(
        "--tmplog",
        action="store_true",
    )

    parser.add_argument(
        "--nr",
        type=int,
        default=-1
    )

    parser.add_argument(
        "--no_wandb",
        action="store_true",
    )

    parser.add_argument(
        "--n_seeds",
        type=int,
        default=None
    )

    parser.add_argument(
        "--num_cmd_groups",
        type=int,
        default=None
    )

    return parser


def make_generator(args, make_generator_helper):
    if args.tmplog or args.debug and args.name is None:
        args.name = "debug"
    else:
        time_str = datetime.datetime.fromtimestamp(time.time()).strftime('%m-%d-')
        args.name = time_str + str(args.name)

    if args.debug or args.tmplog:
        args.no_wandb = True

    if args.wandb_proj_name is not None:
        # prepend data to wandb name
        # time_str = datetime.datetime.fromtimestamp(time.time()).strftime('%m-%d-')
        # args.wandb_proj_name = time_str + args.wandb_proj_name
        pass

    if (args.debug or args.tmplog) and (args.wandb_proj_name is None):
        args.wandb_proj_name = 'debug'

    if not args.debug:
        assert args.name is not None

    # make config generator
    generator = make_generator_helper(args)

    if args.ckpt_mode is None:
        if args.pt:
            args.ckpt_mode = "all"
        else:
            args.ckpt_mode = "best_only"

    set_env_settings(generator, args)
    set_mod_settings(generator, args)
    set_output_dir(generator, args)
    set_num_seeds(generator, args)
    set_wandb_mode(generator, args)

    # set the debug settings last, to override previous setting changes
    set_debug_mode(generator, args)

    """ misc settings """
    generator.add_param(
        key="experiment.validate",
        name="",
        group=-1,
        values=[
            False,
        ],
    )

    # generate jsons and script
    generator.generate(override_base_name=True)
