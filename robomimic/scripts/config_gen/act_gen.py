from robomimic.scripts.config_gen.helper import *

def make_generator_helper(args):
    algo_name_short = "act"
    generator = get_generator(
        algo_name="act",
        config_file=os.path.join(base_path, 'robomimic/exps/templates/act.json'),
        args=args,
        algo_name_short=algo_name_short,
        pt=True,
    )
    if args.ckpt_mode is None:
        args.ckpt_mode = "off"


    generator.add_param(
        key="train.num_epochs",
        name="",
        group=-1,
        values=[1000],
    )

    generator.add_param(
        key="train.batch_size",
        name="",
        group=-1,
        values=[64],
    )

    generator.add_param(
        key="train.max_grad_norm",
        name="",
        group=-1,
        values=[100.0],
    )

    if args.env == "r2d2":
        generator.add_param(
            key="train.data",
            name="ds",
            group=2,
            values=[
                [{"path": p} for p in scan_datasets("~/Downloads/example_pen_in_cup", postfix="trajectory_im128.h5")],
            ],
            value_names=[
                "pen-in-cup",
            ],
        )
        generator.add_param(
            key="train.action_keys",
            name="ac_keys",
            group=-1,
            values=[
                [
                    "action/abs_pos",
                    "action/abs_rot_6d",
                    "action/gripper_position",
                ],
            ],
            value_names=[
                "abs",
            ],
        )
    elif args.env == "kitchen":
        raise NotImplementedError
    elif args.env == "square":
        generator.add_param(
            key="train.data",
            name="ds",
            group=2,
            values=[
                [
                    {"path": "TODO.hdf5"}, # replace with your own path
                ],
            ],
            value_names=[
                "square",
            ],
        )

        # update env config to use absolute action control
        generator.add_param(
            key="experiment.env_meta_update_dict",
            name="",
            group=-1,
            values=[
                {"env_kwargs": {"controller_configs": {"control_delta": False}}}
            ],
        )
        
        generator.add_param(
            key="train.action_keys",
            name="ac_keys",
            group=-1,
            values=[
                [
                    "action_dict/abs_pos",
                    "action_dict/abs_rot_6d",
                    "action_dict/gripper",
                    # "actions",
                ],
            ],
            value_names=[
                "abs",
            ],
        )


    else:
        raise ValueError
    
    generator.add_param(
        key="train.output_dir",
        name="",
        group=-1,
        values=[
            "~/expdata/{env}/{mod}/{algo_name_short}".format(
                env=args.env,
                mod=args.mod,
                algo_name_short=algo_name_short,
            )
        ],
    )

    return generator

if __name__ == "__main__":
    parser = get_argparser()

    args = parser.parse_args()
    make_generator(args, make_generator_helper)
