from robomimic.scripts.config_gen.helper import *

def make_generator_helper(args):
    # basic env settings
    generator = get_generator(
        algo_name="bc_rnn",
        config_file=os.path.join(base_path, 'robomimic/exps/v140_verif/bc_rnn_{}.json'.format(args.mod)),
        args=args,
    )

    generator.add_param(
        key="experiment.logging.log_wandb",
        name="",
        group=-1,
        values=[True],
    )

    generator.add_param(
        key="train.data",
        name="ds",
        group=2,
        values=[
            os.path.join(base_path, 'datasets', args.env, 'ph/image_v140.hdf5')
        ],
        value_names=[
            "{}_v140".format(args.env),
        ],
    )

    
    generator.add_param(
        key="train.output_dir",
        name="",
        group=-1,
        values=[
            "../expdata/{env}/{mod}/bc_rnn".format(
                env=args.env,
                mod=args.mod,
            )
        ],
    )

    generator.add_param(
        key="experiment.save.enabled",
        name="",
        group=-1,
        values=[
            False,
        ],
    )

    return generator


if __name__ == "__main__":
    parser = get_argparser()

    args = parser.parse_args()
    make_generator(args, make_generator_helper)