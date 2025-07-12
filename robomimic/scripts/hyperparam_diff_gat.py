import argparse
import robomimic.utils.hyperparam_utils as HyperparamUtils

def make_gat_sweep_generator_final(config_file, script_file):
    """
    Sets up a final, syntactically correct hyperparameter scan for FlowGAT.

    This version uses explicit parameter values for all generated names,
    as requested.
    """
    generator = HyperparamUtils.ConfigGenerator(
        base_config_file=config_file,
        script_file=script_file,
    )
    generator.add_param(
        key="experiment.logging.wandb_proj_name",
        name="wandb_name",
        group="1",
        values=["General_Model_Obs_Hist"],
        value_names=["General_Model_Obs_Hist"]
    )
    
    generator.add_param(
        key="train.frame_stack",
        name="obs_hist",
        group="2",
        values=[1, 2, 5, 10],
        value_names=["1", "2", "5", "10"]
    )
    
    generator.add_param(
        key="algo.temp_edges",
        name="temp_edges",
        group="3",
        values=[True, False],
        value_names=["True", "False"]
    )
    
    # --- Group 5: Seed ---
    generator.add_param(
        key="train.seed",
        name="seed",
        group=4,
        values=[0, 25, 42],
        value_names=["0", "25", "42"] # Use values directly
    )

    return generator


def main(args):
    """
    Generates the configuration files and the execution script.
    """
    generator = make_gat_sweep_generator_final(config_file=args.config, script_file=args.script)
    generator.generate()

    print(f"Generated script file: {args.script}")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a final, correct hyperparameter sweep script for FlowGAT with explicit naming."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the base FlowGAT JSON config file.",
    )
    parser.add_argument(
        "--script",
        type=str,
        default="run_sweep_explicit.sh",
        help="Path to the output shell script.",
    )
    args = parser.parse_args()
    main(args)
