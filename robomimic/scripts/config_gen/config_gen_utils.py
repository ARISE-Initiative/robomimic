import argparse
import os
import time
import datetime

import robomimic
import robomimic.macros as macros
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

    return generator


def set_env_settings(generator, args):
    if args.env == "robocasa":
        generator.add_param(
            key="train.action_config",
            name="",
            group=-1,
            values=[
                {
                    "actions":{
                        "normalization": None,
                    },
                    "actions_abs":{
                        "normalization": "min_max",
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

        # language conditioned architecture
        generator.add_param(
            key="observation.encoder.rgb.core_class",
            name="",
            group=-1,
            values=[
                "VisualCoreLanguageConditioned"
            ],
        )
        generator.add_param(
            key="observation.encoder.rgb.core_kwargs.backbone_class",
            name="",
            group=-1,
            values=[
                "ResNet18ConvFiLM"
            ],
        )

        env_kwargs = {
            "generative_textures": None,
            "scene_split": None,
            "style_ids": None,
            "layout_ids": None,
            "layout_and_style_ids": [[1, 1], [2, 2], [4, 4], [6, 9], [7, 10]],
            "randomize_cameras": False,
            "obj_instance_split": "B",
        }
        if args.abs_actions:
            env_kwargs["controller_configs"] = {"control_delta": False}
            generator.add_param(
                key="train.action_keys",
                name="ac_keys",
                group=-1,
                values=[
                    [
                        "actions_abs",
                    ],
                ],
                value_names=[
                    "abs_acs",
                ],
                hidename=True,
            )

        # don't use generative textures for evaluation
        generator.add_param(
            key="experiment.env_meta_update_dict",
            name="",
            group=-1,
            values=[{"env_kwargs": env_kwargs}],
        )

        generator.add_param(
            key="observation.encoder.rgb.obs_randomizer_kwargs",
            name="obsrandargs",
            group=-1,
            values=[
                {"crop_height": 116, "crop_width": 116, "num_crops": 1, "pos_enc": False},
            ],
            hidename=True,
        )
        if "experiment.rollout.n" not in generator.parameters:
            generator.add_param(
                key="experiment.rollout.n",
                name="",
                group=-1,
                values=[50],
                value_names=[""],
            )
        generator.add_param(
            key="experiment.rollout.horizon",
            name="",
            group=-1,
            values=[500],
            value_names=[""],
        )
        if args.mod == 'im':
            generator.add_param(
                key="observation.modalities.obs.low_dim",
                name="",
                group=-1,
                values=[
                    ["robot0_base_to_eef_pos",
                     "robot0_base_to_eef_quat",
                     "robot0_base_pos",
                     "robot0_base_quat",
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
                values=[100],
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
                values=[5],
            )
        generator.add_param(
            key="train.hdf5_cache_mode",
            name="",
            group=-1,
            # values=["low_dim"],
            values=[None],
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
                values=[1000],
            )
        if "experiment.rollout.rate" not in generator.parameters:
            generator.add_param(
                key="experiment.rollout.rate",
                name="",
                group=-1,
                values=[100],
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
    
    # set horizon to 30
    ds_cfg_list = generator.parameters["train.data"].values
    for ds_cfg in ds_cfg_list:
        for d in ds_cfg:
            d["horizon"] = 30
    
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
        values=[2], #50
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
        # values=["low_dim"],
        values=[None],
        value_names=[""],
    )
    generator.add_param(
        key="train.num_data_workers",
        name="",
        group=-1,
        values=[3],
    )


def get_output_dir(args, algo_dir):
    return "{expdata_base_path}/{env}/{algo_dir}".format(
        expdata_base_path=get_expdata_base_path(),
        env=args.env,
        algo_dir=algo_dir,
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

def set_rollout_mode(generator, args):
    if args.no_rollout:
        generator.add_param(
            key="experiment.rollout.enabled",
            name="",
            group=-1,
            values=[False],
        )


def set_num_seeds(generator, args):
    if "train.seed" not in generator.parameters:
        if args.n_seeds is not None:
            generator.add_param(
                key="train.seed",
                name="seed",
                group=-10,
                values=[i + 1 for i in range(args.n_seeds)],
                prepend=True,
            )
        else:
            generator.add_param(
                key="train.seed",
                name="seed",
                group=-10,
                values=[123],
                prepend=True,
            )


def get_expdata_base_path():
    expdata_base_path = macros.EXPDATA_BASE_PATH
    if expdata_base_path is None:
        expdata_base_path = os.path.join(base_path, "expdata")
    return expdata_base_path


def get_robocasa_ds(
        ds_names,
        exclude_ds_names=None,
        src="human",
        filter_key=None,
        eval=None
    ):
    from robocasa.utils.dataset_registry import get_ds_path, SINGLE_STAGE_TASK_DATASETS, MULTI_STAGE_TASK_DATASETS

    assert src in ["human", "mg"]

    all_datasets = {}
    all_datasets.update(SINGLE_STAGE_TASK_DATASETS)
    all_datasets.update(MULTI_STAGE_TASK_DATASETS)

    if ds_names == "all":
        ds_names = list(all_datasets.keys())
    elif ds_names == "single_stage":
        ds_names = list(SINGLE_STAGE_TASK_DATASETS.keys())
    elif ds_names == "multi_stage":
        ds_names = list(MULTI_STAGE_TASK_DATASETS.keys())
    elif isinstance(ds_names, str):
        ds_names = [ds_names]

    if exclude_ds_names is not None:
        ds_names = [name for name in ds_names if name not in exclude_ds_names]

    ret = []
    for name in ds_names:
        cfg = dict()
        ds_path = get_ds_path(name, ds_type=f"{src}_im")

        # set path and horizon
        cfg["path"] = ds_path
        cfg["horizon"] = all_datasets[name]["horizon"]
        
        # determine whether we are performing eval on dataset
        if eval is None or name in eval:
            cfg["do_eval"] = True
        else:
            cfg["do_eval"] = False

        # determine dataset filter key
        if filter_key is not None:
            cfg["filter_key"] = filter_key
        else:
            if src == "human":
                cfg["filter_key"] = "50_demos"
            elif src == "mg":
                cfg["filter_key"] = "3000_demos"

        ret.append(cfg)

    return ret


def get_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--name",
        type=str,
    )

    parser.add_argument(
        "--env",
        type=str,
        default='robocasa',
    )

    parser.add_argument(
        '--mod',
        type=str,
        choices=['ld', 'im'],
        default='im',
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
        '--no_rollout',
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
        "--abs_actions",
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


def make_generator(args, make_generator_helper, skip_helpers=None):
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

    if skip_helpers is None:
        skip_helpers = []

    if "env" not in skip_helpers:
        set_env_settings(generator, args)
    if "mod" not in skip_helpers:
        set_mod_settings(generator, args)
    set_output_dir(generator, args)
    set_num_seeds(generator, args)
    set_wandb_mode(generator, args)

    # set the debug settings last, to override previous setting changes
    set_debug_mode(generator, args)

    set_rollout_mode(generator, args)

    """ misc settings """
    generator.add_param(
        key="experiment.validate",
        name="",
        group=-1,
        values=[
            False,
        ],
    )
    if "experiment.save.on_best_rollout_success_rate" not in generator.parameters:
        generator.add_param(
            key="experiment.save.on_best_rollout_success_rate",
            name="",
            group=-1,
            values=[
                False,
            ],
        )

    # generate jsons and script
    generator.generate(override_base_name=True)