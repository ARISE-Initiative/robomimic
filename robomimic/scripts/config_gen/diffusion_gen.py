from robomimic.scripts.config_gen.helper import *

def make_generator_helper(args):
    algo_name_short = "diffusion_policy"

    args.abs_actions = True

    generator = get_generator(
        algo_name="diffusion_policy",
        config_file=os.path.join(base_path, 'robomimic/exps/templates/diffusion_policy.json'),
        args=args,
        algo_name_short=algo_name_short,
        pt=True,
    )
    if args.ckpt_mode is None:
        args.ckpt_mode = "off"

    generator.add_param(
        key="train.num_data_workers",
        name="",
        group=-1,
        values=[8],
    )

    generator.add_param(
        key="train.num_epochs",
        name="",
        group=-1,
        values=[1000],
    )

    # use ddim by default
    generator.add_param(
        key="algo.ddim.enabled",
        name="ddim",
        group=1001,
        values=[
            True,
            # False,
        ],
        hidename=True,
    )
    generator.add_param(
        key="algo.ddpm.enabled",
        name="ddpm",
        group=1001,
        values=[
            False,
            # True,
        ],
        hidename=True,
    )

    if args.env == "robocasa":
        generator.add_param(
            key="train.data",
            name="ds",
            group=2,
            values_and_names=[
                ([{
                    "horizon": 500,
                    "do_eval": True,
                    "filter_key": "1000_demos",
                    "path": "/data1/aaronl/spark/bare/mg/2024-03-25-05-52-19/demo3_gentex_im128_randcams.hdf5"
                }], "test"),
                # (get_ds_cfg(["PnPCounterToSink"], src="mg", eval=None), "mg"),
                # (get_ds_cfg("all", ds_repo="human"), "pnp-doors-human"),
                # (get_ds_cfg("pnp_cab_to_counter", ds_repo="human", filter_key="100_demos"), "pnp-cab-to-counter-human-100"),
                # (get_ds_cfg("pnp_cab_to_counter", ds_repo="mg", filter_key="100_demos"), "pnp-cab-to-counter-mg-100"),
                # (get_ds_cfg("pnp_cab_to_counter", ds_repo="mg", filter_key="1000_demos"), "pnp-cab-to-counter-mg-1000"),
                # (get_ds_cfg("pnp_cab_to_counter", ds_repo="mg", filter_key="5000_demos"), "pnp-cab-to-counter-mg-5000"),

                # (get_ds_cfg("all", ds_repo="mg", filter_key="1000_demos"), "4-pnp-tasks-mg-1000"),
                # (get_ds_cfg("all", ds_repo="mg", filter_key="5000_demos"), "4-pnp-tasks-mg-5000"),
            ]
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
                    "action_dict/base_mode",
                    # "actions",
                ],
            ],
            value_names=[
                "abs",
            ],
            hidename=True,
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