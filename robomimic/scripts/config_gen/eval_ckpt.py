from robomimic.scripts.config_gen.config_gen_utils import *
import json


def make_generator_helper(args):
    ckpt_path = args.ckpt

    # get ckpt config file, infer algo name from path
    ckpt_config_path = os.path.join(os.path.dirname(ckpt_path), "../config.json")
    with open(ckpt_config_path) as f:
        ckpt_config = json.load(f)
    algo_name_short = ckpt_path.split("/")[-6]

    generator = get_generator(
        algo_name=ckpt_config["algo_name"],
        config_file=ckpt_config_path,
        args=args,
        algo_name_short=algo_name_short,
    )

    # set up datasets to evaluate on
    ckpt_datasets = ckpt_config["train"]["data"]
    for ds_cfg in ckpt_datasets:
        ds_cfg["eval"] = True
    generator.add_param(
        key="train.data",
        name="ds",
        group=12345,
        values_and_names=[
            (ckpt_datasets, "ckpt_datasets"),
            (get_robocasa_ds("single_stage", src="human", eval=["PnPCounterToSink", "PnPCounterToCab"], filter_key="50_demos"), "human-50"), # training on human datasets
            # ("set-your-datasets-here"), "name"),
        ],
    )

    # set up configs for running evals (do not need to change these lines)
    generator.add_param(
        key="experiment.ckpt_path",
        name="ckpt",
        group=1,
        values=[ckpt_path],
        hidename=True,
    )
    generator.add_param(
        key="experiment.rollout.warmstart",
        name="",
        group=-1,
        values=[-1],
    )
    generator.add_param(
        key="train.num_epochs",
        name="",
        group=-1,
        values=[0],
    )
    generator.add_param(
        key="train.num_data_workers",
        name="",
        group=-1,
        values=[0],
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

    parser.add_argument(
        "--ckpt",
        type=str,
        help="path to model checkpoint (must be *.pth file)",
        required=True,
    )

    args = parser.parse_args()
    make_generator(args, make_generator_helper, skip_helpers=("env", "mod"))
