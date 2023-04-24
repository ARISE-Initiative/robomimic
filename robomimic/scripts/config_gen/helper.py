import argparse
import os
import time
import datetime

import robomimic
import robomimic.utils.hyperparam_utils as HyperparamUtils

base_path = os.path.abspath(os.path.join(os.path.dirname(robomimic.__file__), os.pardir))

def get_generator(algo_name, config_file, args, algo_name_short=None):
    if args.wandb_proj_name is None:
        strings = [
            algo_name_short if (algo_name_short is not None) else algo_name,
            args.name,
            args.env,
            args.mod,
        ]
        args.wandb_proj_name = '_'.join([s for s in strings if s is not None])

    if args.script is not None:
        generated_config_dir = os.path.join(os.path.dirname(args.script), "json")
    else:
        curr_time = datetime.datetime.fromtimestamp(time.time()).strftime('%m-%d-%y-%H-%M-%S')
        generated_config_dir=os.path.join(
            '~/', 'tmp/autogen_configs/il', algo_name, args.env, args.mod, args.name, curr_time, "json",
        )

    generator = HyperparamUtils.ConfigGenerator(
        base_config_file=config_file,
        generated_config_dir=generated_config_dir,
        wandb_proj_name=args.wandb_proj_name,
        script_file=args.script,
    )

    return generator

def set_debug_mode(generator, args):
    if not args.debug:
        return

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
    generator.add_param(
        key="experiment.name",
        name="",
        group=-1,
        values=["debug"],
        value_names=[""],
    )
    generator.add_param(
        key="train.batch_size",
        name="",
        group=-1,
        values=[16],
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
        # values=[None],
        # value_names=[""],

        values=["low_dim"],
    )

def set_exp_id(generator, args):
    assert args.name is not None

    vals = generator.parameters["train.output_dir"].values

    for i in range(len(vals)):
        vals[i] = os.path.join(vals[i], args.name)

def set_num_rollouts(generator, args):
    if args.nr < 0:
        return
    generator.add_param(
        key="experiment.rollout.n",
        name="",
        group=-1,
        values=[args.nr],
        value_names=[""],
    )

def set_wandb_mode(generator, args):
    if args.no_wandb:
        generator.add_param(
            key="experiment.logging.log_wandb",
            name="",
            group=-1,
            values=[False],
        )

def set_video_mode(generator, args):
    if args.no_video:
        generator.add_param(
            key="experiment.render_video",
            name="",
            group=-1,
            values=[False],
        )
        generator.add_param(
            key="experiment.keep_all_videos",
            name="",
            group=-1,
            values=[False],
        )
    else:
        generator.add_param(
            key="experiment.keep_all_videos",
            name="",
            group=-1,
            values=[True],
        )

def set_ckpt_mode(generator, args):
    if args.save_ckpts:
        generator.add_param(
            key="experiment.save.enabled",
            name="",
            group=-1,
            values=[True],
        )

def set_num_seeds(generator, args):
    if args.n_seeds is not None:
        generator.add_param(
            key="train.seed",
            name="seed",
            group=-10,
            values=[i + 1 for i in range(args.n_seeds)],
        )

def set_cuda_mode(generator, args):
    if args.no_cuda:
        generator.add_param(
            key="train.cuda",
            name="",
            group=-1,
            values=[False],
            # hidename=True,
        )


def set_rollout_mode(generator, args):
    if args.no_rollout:
        generator.add_param(
            key="experiment.rollout.enabled",
            name="",
            group=-1,
            values=[False],
            # hidename=True,
        )


def set_env_settings(generator, args):
    if args.env == 'calvin':
        if "observation.modalities.obs.low_dim" not in generator.parameters:
            if args.mod == 'ld':
                generator.add_param(
                    key="observation.modalities.obs.low_dim",
                    name="lowdimkeys",
                    group=-1,
                    values=[
                        ["scene_obs", "env_id", "robot0_eef_pos", "robot0_eef_euler", "robot0_gripper_qpos"],
                    ],
                    value_names=[
                        "scene_proprio",
                    ],
                    # hidename=True,
                )
            else:
                generator.add_param(
                    key="observation.modalities.obs.low_dim",
                    name="lowdimkeys",
                    group=-1,
                    values=[
                        ["robot0_eef_pos", "robot0_eef_euler", "robot0_gripper_qpos"],
                    ],
                    value_names=[
                        "proprio",
                    ],
                    # hidename=True,
                )

        if args.mod == 'im':
            generator.add_param(
                key="observation.modalities.obs.rgb",
                name="",
                group=-1,
                values=[["rgb_static"]],
            )
            generator.add_param(
                key="observation.modalities.obs.rgb2",
                name="",
                group=-1,
                values=[["rgb_gripper"]],
            )

        if "experiment.rollout.horizon" not in generator.parameters:
            generator.add_param(
                key="experiment.rollout.horizon",
                name="",
                group=-1,
                values=[1000],
            )
    elif args.env == 'kitchen':
        generator.add_param(
            key="observation.modalities.obs.low_dim",
            name="",
            group=-1,
            values=[
                ["flat"]
            ],
        )
        if args.mod == 'im':
            generator.add_param(
                key="observation.modalities.obs.low_dim",
                name="",
                group=-1,
                values=[
                    ["robot_joints"]
                ],
            )
            generator.add_param(
                key="observation.modalities.obs.rgb",
                name="",
                group=-1,
                values=[
                    ["agentview_image", "eye_in_hand_image"]
                ],
            )

        if "experiment.rollout.horizon" not in generator.parameters:
            generator.add_param(
                key="experiment.rollout.horizon",
                name="",
                group=-1,
                values=[280],
            )
    elif args.env in ['square', 'lift', 'can']:
        # # set videos off
        # args.no_video = True

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
    elif args.env in ['real_breakfast', 'real_lift', 'real_cook']:
        assert args.mod == "im"
        generator.add_param(
            key="experiment.save.enabled",
            name="",
            group=-1,
            values=[
                True
            ],
        )
        generator.add_param(
            key="experiment.rollout.enabled",
            name="",
            group=-1,
            values=[
                False
            ],
        )
        generator.add_param(
            key="observation.modalities.obs.low_dim",
            name="",
            group=-1,
            values=[
                ["ee_pos", "ee_quat", "gripper_states"]
            ],
        )
        generator.add_param(
            key="observation.modalities.obs.rgb",
            name="",
            group=-1,
            values=[
                ["agentview_rgb", "eye_in_hand_rgb"]
            ],
        )
    else:
        raise NotImplementedError

def set_mod_settings(generator, args):
    return
    
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
        generator.add_param(
            key="experiment.rollout.epochs",
            name="",
            group=-1,
            values=[
                [50, 100, 150] + [100*i for i in range(2, 21)],
            ],
        )
    elif args.mod == 'im':        
        if "experiment.save.epochs" not in generator.parameters:
            generator.add_param(
                key="experiment.save.epochs",
                name="",
                group=-1,
                values=[
                    [200, 400, 600]
                ],
            )
        generator.add_param(
            key="experiment.epoch_every_n_steps",
            name="",
            group=-1,
            values=[500],
        )
        generator.add_param(
            key="train.num_data_workers",
            name="",
            group=-1,
            values=[2],
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
            if args.env == 'real_lift':
                generator.add_param(
                    key="train.num_epochs",
                    name="",
                    group=-1,
                    values=[200],
                )
            elif args.env == 'real_breakfast':
                generator.add_param(
                    key="train.num_epochs",
                    name="",
                    group=-1,
                    values=[600],
                )
            else:
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

def set_r3m_mode(generator, args):
    if not args.r3m:
        return
    
    assert args.mod == 'im'

    generator.add_param(
        key="observation.encoder.rgb.core_kwargs.backbone_kwargs.pretrained",
        name="pretrained",
        group=-1,
        values=[
            'r3m',
        ],
        # hidename=True,
    )
    generator.add_param(
        key="observation.encoder.rgb.core_kwargs.backbone_kwargs.freeze",
        name="freeze",
        group=-1,
        values_and_names=[
            (True, "T"),
        ],
        # hidename=True,
    )
    generator.add_param(
        key="observation.encoder.rgb2.core_kwargs.backbone_kwargs.pretrained",
        name="",
        group=-1,
        values=[
            'r3m',
        ],
        # hidename=True,
    )
    generator.add_param(
        key="observation.encoder.rgb2.core_kwargs.backbone_kwargs.freeze",
        name="",
        group=-1,
        values_and_names=[
            (True, "T"),
        ],
        # hidename=True,
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
        default='calvin',
    )

    parser.add_argument(
        '--mod',
        type=str,
        choices=['ld', 'im'],
        # default='ld',
        required=True,
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
        "--save_ckpts",
        action="store_true",
    )

    parser.add_argument(
        "--no_cuda",
        action="store_true",
    )

    parser.add_argument(
        "--no_rollout",
        action="store_true",
    )

    parser.add_argument(
        "--r3m",
        action="store_true",
    )

    parser.add_argument(
        "--n_seeds",
        type=int,
        default=None
    )

    return parser

def make_generator(args, make_generator_helper):
    if args.tmplog or args.debug:
        args.name = "debug"
    else:
        time_str = datetime.datetime.fromtimestamp(time.time()).strftime('%m-%d-')
        args.name = time_str + args.name

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

    set_env_settings(generator, args)
    set_mod_settings(generator, args)
    set_debug_mode(generator, args)
    set_num_rollouts(generator, args)
    set_exp_id(generator, args)
    set_wandb_mode(generator, args)
    set_video_mode(generator, args)
    set_ckpt_mode(generator, args)
    set_num_seeds(generator, args)
    set_cuda_mode(generator, args)
    set_rollout_mode(generator, args)
    set_r3m_mode(generator, args)

    # """ misc settings """
    # generator.add_param(
    #     key="experiment.save.on_best_rollout_success_rate",
    #     name="",
    #     group=-1,
    #     values=[
    #         False
    #     ],
    # )
    # generator.add_param(
    #     key="train.load_next_obs",
    #     name="",
    #     group=-1,
    #     values=[False],
    # )

    # generate jsons and script
    generator.generate()