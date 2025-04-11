import argparse
import robomimic
import robomimic.utils.hyperparam_utils as HyperparamUtils

# Define the corrected function for the small sweep
def make_generator_simple(config_file, script_file):
    """
    Sets up a *simplified* hyperparameter scan (~12 configs) for DiffGAT,
    focusing on tradeoff, learning rate, and model size, respecting group rules.
    """
    generator = HyperparamUtils.ConfigGenerator(
        base_config_file=config_file,
        script_file=script_file,
    )

    # --- Base Settings ---
    generator.add_param(
        key="experiment.logging.log_wandb",
        name="",
        group=-1,
        values=[False], # Ensure WandB logging is enabled
    )
    generator.add_param(
        key="experiment.logging.wandb_proj_name",
        name="",
        group=-1,
        values=["robomimic"], # Ensure TensorBoard logging is enabled
    )

    # --- Simplified Hyperparameter Sweep ---
    # Parameters are placed in different groups because they have different numbers of values.

    # Group 1: Loss Tradeoff (3 Values) - Highest Priority
    generator.add_param(
        key="algo.loss.tradeoff",
        name="tradeoff",
        group=1,
        values=[0.5, 0.8, 1.0], # Balanced, favoring noise, only noise
        value_names=["0p5", "0p8", "1p0"]
    )

    # Group 2: Learning Rate (2 Values) - Second Priority
    generator.add_param(
        key="algo.optim_params.policy.learning_rate.initial",
        name="lr",
        group=2,
        values=[1e-4, 5e-5], # Default-ish and lower
        value_names=["1e-4", "5e-5"]
    )

    # Group 3: Model Size (GNN Hidden Dimension) (2 Values) - Third Priority
    generator.add_param(
        key="algo.gnn.hidden_dim",
        name="gnnd",
        group=3,
        values=[256, 512], # Smaller and medium capacity
        value_names=["256", "512"]
    )

    # NOTE: All other parameters will use values from the base config.

    return generator


def main(args):
    """
    Generates the configuration files and the execution script.
    Uses the corrected simplified generator.
    """
    # Use the simplified generator function
    generator = make_generator_simple(config_file=args.config, script_file=args.script)

    # Generate the json files and the shell script
    generator.generate()

    print(f"Generated script file: {args.script}")
    print("\nInstructions:")
    print(f"1. Ensure your base config ('{args.config}') is correct.")
    print(f"2. This script sweeps tradeoff (3 vals), learning rate (2 vals), and hidden_dim (2 vals).")
    print(f"3. Execute the generated script to run the hyperparameter scan: bash {args.script}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate simplified hyperparameter scan configs and script for DiffGAT."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the base DiffGAT JSON config file. New configs will be generated in the same directory.",
    )
    parser.add_argument(
        "--script",
        type=str,
        required=True,
        help="Path to the output shell script that will contain commands to run all generated training runs.",
    )
    args = parser.parse_args()
    main(args)