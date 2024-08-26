from robomimic.scripts.config_gen.config_gen_utils import *


def make_generator_helper(args):
    algo_name_short = "bc_rnn"
    generator = get_generator(
        algo_name="bc",
        config_file=os.path.join(base_path, 'robomimic/exps/templates/bc.json'),
        args=args,
        algo_name_short=algo_name_short,
    )

    if args.env == "robocasa":
        raise NotImplementedError
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
        values=[get_output_dir(args, algo_dir=algo_name_short)]
    )

    return generator

if __name__ == "__main__":
    parser = get_argparser()

    args = parser.parse_args()
    make_generator(args, make_generator_helper)
