from robomimic.scripts.config_gen.config_gen_utils import *


def make_generator_helper(args):
    algo_name_short = "act"
    generator = get_generator(
        algo_name="act",
        config_file=os.path.join(base_path, 'robomimic/exps/templates/act.json'),
        args=args,
        algo_name_short=algo_name_short,
    )

    if args.env == "robocasa":
        raise NotImplementedError
    else:
        raise ValueError
    
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
