from robomimic.scripts.config_gen.helper import *

def make_generator_helper(args):
    algo_name_short = "bc_rnn"

    generator = get_generator(
        algo_name="bc",
        config_file=os.path.join(base_path, 'robomimic/exps/templates/bc.json'),
        args=args,
        algo_name_short=algo_name_short,
        pt=True,
    )
    if args.ckpt_mode is None:
        args.ckpt_mode = "off"

    if args.env == "r2d2":
        raise NotImplementedError
    elif args.env == "kitchen":
        generator.add_param(
            key="train.data",
            name="ds",
            group=2,
            values=[
                [{"path": "~/datasets/kitchen/prior/human_demos/pnp_table_to_cab/bowls/20230816_im84.hdf5", "filter_key": "100_demos"}],
                # [{"path": "~/datasets/kitchen/prior/human_demos/pnp_table_to_cab/all/20230806_im84.hdf5", "filter_key": "100_demos"}],
                # [{"path": "~/datasets/kitchen/prior/mimicgen/pnp_table_to_cab/viraj_mg_2023-08-10-20-31-14/demo_im84.hdf5", "filter_key": "100_demos"}],
                # [{"path": "~/datasets/kitchen/prior/mimicgen/pnp_table_to_cab/viraj_mg_2023-08-10-20-31-14/demo_im84.hdf5", "filter_key": "1000_demos"}],
            ],
            value_names=[
                "bowls-human-100",
                # "human-100",
                # "mg-100",
                # "mg-1000",
            ],
        )
    else:
        raise ValueError

    # change default settings: rnn, predict 10 steps into future
    generator.add_param(
        key="algo.rnn.enabled",
        name="",
        group=-1,
        values=[True],
        hidename=True,
    )
    generator.add_param(
        key="train.seq_length",
        name="",
        group=-1,
        values=[10],
        hidename=True,
    )
    generator.add_param(
        key="algo.rnn.horizon",
        name="",
        group=-1,
        values=[10],
        hidename=True,
    )
    if args.mod == "im":
        generator.add_param(
            key="algo.rnn.hidden_dim",
            name="",
            group=-1,
            values=[1000],
            hidename=True,
        )

    generator.add_param(
        key="algo.gmm.enabled",
        name="gmm",
        group=130801,
        values=[True],
    )
    generator.add_param(
        key="algo.gmm.min_std",
        name="mindstd",
        group=271314,
        values=[
            0.03,
            #0.0001,
        ],
        hidename=True,
    )
    generator.add_param(
        key="train.max_grad_norm",
        name="maxgradnorm",
        group=18371,
        values=[
            # None,
            100.0,
        ],
        hidename=True,
    )
    
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
