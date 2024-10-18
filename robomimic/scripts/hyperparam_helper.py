"""
A useful script for generating json files and shell scripts for conducting parameter scans.
The script takes a path to a base json file as an argument and a shell file name.
It generates a set of new json files in the same folder as the base json file, and 
a shell file script that contains commands to run for each experiment.

Instructions:

(1) Start with a base json that specifies a complete set of parameters for a single 
    run. This only needs to include parameters you want to sweep over, and parameters
    that are different from the defaults. You can set this file path by either
    passing it as an argument (e.g. --config /path/to/base.json) or by directly
    setting the config file in @make_generator. The new experiment jsons will be put
    into the same directory as the base json.

(2) Decide on what json parameters you would like to sweep over, and fill those in as 
    keys in @make_generator below, taking note of the hierarchical key
    formatting using "/" or ".". Fill in corresponding values for each - these will
    be used in creating the experiment names, and for determining the range
    of values to sweep. Parameters that should be sweeped together should
    be assigned the same group number.

(3) Set the output script name by either passing it as an argument (e.g. --script /path/to/script.sh)
    or by directly setting the script file in @make_generator. The script to run all experiments
    will be created at the specified path.

Args:
    config (str): path to a base config json file that will be modified to generate config jsons.
        The jsons will be generated in the same folder as this file.

    script (str): path to output script that contains commands to run the generated training runs

Example usage:

    # assumes that /tmp/gen_configs/base.json has already been created (see quickstart section of docs for an example)
    python hyperparam_helper.py --config /tmp/gen_configs/base.json --script /tmp/gen_configs/out.sh
"""
import argparse

import robomimic
import robomimic.utils.hyperparam_utils as HyperparamUtils


def make_generator(config_file, script_file):
    """
    Implement this function to setup your own hyperparameter scan!
    """
    generator = HyperparamUtils.ConfigGenerator(
        base_config_file=config_file, script_file=script_file
    )

    # use RNN with horizon 10
    generator.add_param(
        key="algo.rnn.enabled",
        name="", 
        group=0, 
        values=[True],
    )
    generator.add_param(
        key="train.seq_length", 
        name="", 
        group=0, 
        values=[10], 
    )
    generator.add_param(
        key="algo.rnn.horizon",
        name="", 
        group=0, 
        values=[10], 
    )

    # LR - 1e-3, 1e-4
    generator.add_param(
        key="algo.optim_params.policy.learning_rate.initial", 
        name="plr", 
        group=1, 
        values=[1e-3, 1e-4], 
    )

    # GMM y / n
    generator.add_param(
        key="algo.gmm.enabled", 
        name="gmm", 
        group=2, 
        values=[True, False], 
        value_names=["t", "f"],
    )

    # RNN dim 400 + MLP dims (1024, 1024) vs. RNN dim 1000 + empty MLP dims ()
    generator.add_param(
        key="algo.rnn.hidden_dim", 
        name="rnnd", 
        group=3, 
        values=[
            400, 
            1000,
        ], 
    )
    generator.add_param(
        key="algo.actor_layer_dims", 
        name="mlp", 
        group=3, 
        values=[
            [1024, 1024], 
            [],
        ], 
        value_names=["1024", "0"],
    )

    return generator


def main(args):

    # make config generator
    generator = make_generator(config_file=args.config, script_file=args.script)

    # generate jsons and script
    generator.generate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Path to base json config - will override any defaults.
    parser.add_argument(
        "--config",
        type=str,
        help="path to base config json that will be modified to generate jsons. The jsons will\
            be generated in the same folder as this file.",
    )

    # Script name to generate - will override any defaults
    parser.add_argument(
        "--script",
        type=str,
        help="path to output script that contains commands to run the generated training runs",
    )

    args = parser.parse_args()
    main(args)
