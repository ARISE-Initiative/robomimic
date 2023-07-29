from robomimic.scripts.config_gen.helper import *

def make_generator_helper(args):
    algo_name_short = "bc_xfmr"

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
        values=[4],
    )
    generator.add_param(
        key="experiment.save.every_n_epochs",
        name="",
        group=-1,
        values=[
            100
        ],
    )

    # run rollouts at epoch 0 only
    generator.add_param(
        key="experiment.rollout.warmstart",
        name="",
        group=-1,
        values=[
            -1,
        ],
    )
    generator.add_param(
        key="train.num_epochs",
        name="",
        group=-1,
        values=[40],
    )
    generator.add_param(
        key="experiment.rollout.rate",
        name="",
        group=-1,
        values=[10],
    )

    if args.env == "r2d2":
        generator.add_param(
            key="train.data",
            name="ds",
            group=2,
            values=[
                # [{"path": p} for p in scan_datasets("~/code/r2d2/data/success/2023-05-23_t2c-cans", postfix="trajectory_im84.h5")],
                [{"path": p} for p in scan_datasets("/home/cchi/local/data/r2d2/pen/success/2023-02-28", postfix="trajectory_im128.h5")],
            ],
            value_names=[
                "pnp-t2c-cans-84",
                # "pnp-t2c-cans-128",
            ],
        )
        generator.add_param(
            key="observation.encoder.rgb.obs_randomizer_kwargs.crop_height",
            name="",
            group=2,
            values=[
                76,
                # 116
            ],
        )
        generator.add_param(
            key="observation.encoder.rgb.obs_randomizer_kwargs.crop_width",
            name="",
            group=2,
            values=[
                76,
                # 116
            ],
        )
    elif args.env == "square":
        generator.add_param(
            key="train.data",
            name="ds",
            group=2,
            values=[
                [
                    {"path": "~/datasets/square/ph/image_v141.hdf5"},
                    {"path": "~/datasets/square/ph/image_v141.hdf5"},
                ],
            ],
            value_names=[
                "square",
            ],
        )
    else:
        raise ValueError

    if "experiment.ckpt_path" in generator.parameters:
        generator.add_param(
            key="algo.optim_params.policy.learning_rate.initial",
            name="lrinit",
            group=110,
            values=[
                1e-5,
            ],
            hidename=True,
        )
        generator.add_param(
            key="algo.optim_params.policy.learning_rate.lr_scheduler_type",
            name="lrsch",
            group=111,
            values=[
                # "linear",
                None,
            ],
            value_names=[
                "none"
            ],
            hidename=True,
        )
    
    generator.add_param(
        key="train.output_dir",
        name="",
        group=-1,
        values=[
            "/home/cchi/dev/robomimic_r2d2/datasets/experiment_results/{env}/{mod}/{algo_name_short}".format(
                env=args.env,
                mod=args.mod,
                algo_name_short=algo_name_short,
            )
        ],
    )

    generator.add_param(
        key="experiment.rollout.enabled",
        name="",
        group=-1,
        values=[
            True
        ],
        hidename=False,
    )

    return generator

if __name__ == "__main__":
    parser = get_argparser()

    args = parser.parse_args()
    make_generator(args, make_generator_helper)