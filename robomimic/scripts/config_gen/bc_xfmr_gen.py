from robomimic.scripts.config_gen.config_gen_utils import *


def make_generator_helper(args):
    algo_name_short = "bc_xfmr"

    generator = get_generator(
        algo_name="bc",
        config_file=os.path.join(base_path, 'robomimic/exps/templates/bc_transformer.json'),
        args=args,
        algo_name_short=algo_name_short,
    )

    ### Define dataset variants to train on ###
    generator.add_param(
        key="train.data",
        name="ds",
        group=123456,
        values_and_names=[
            (get_robocasa_ds("single_stage", src="human", eval=["PnPCounterToSink", "PnPCounterToCab"], filter_key="50_demos"), "human-50"), # training on human datasets
            (get_robocasa_ds("single_stage", src="mg", eval=["PnPCounterToSink", "PnPCounterToCab"], filter_key="3000_demos"), "mg-3000"), # training on MimicGen datasets

            # composite tasks
            (get_robocasa_ds("ArrangeVegetables", filter_key="50_demos"), "ArrangeVegetables"),
            (get_robocasa_ds("MicrowaveThawing", filter_key="50_demos"), "MicrowaveThawing"),
            (get_robocasa_ds("RestockPantry", filter_key="50_demos"), "RestockPantry"),
            (get_robocasa_ds("PreSoakPan", filter_key="50_demos"), "PreSoakPan"),
            (get_robocasa_ds("PrepareCoffee", filter_key="50_demos"), "PrepareCoffee"),
        ]
    )

    """
    ### Uncomment this code to fine-tune on existing checkpoint ###
    generator.add_param(
        key="experiment.ckpt_path",
        name="ckpt",
        group=1389,
        values_and_names=[
            (None, "none"),
            # ("set checkpoint pth path here", "trained-ckpt"),
        ],
    )
    """

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
