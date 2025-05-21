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

    # --- Simplified Hyperparameter Sweep ---
    # Parameters are placed in different groups because they have different numbers of values.

    # Group 1: Loss Tradeoff (3 Values) - Highest Priority
    generator.add_param(
        key="train.seed",
        name="seed",
        group=2,
        values=[0,25,42],
        value_names=["0", "25", "42"]
    )
    generator.add_param(
        key ="algo.optim_params.policy.learning_rate.scheduler_type",
        name="lr_scheduler_type",
        group=1,
        values=["cosine_restart"],
        value_names=["cosine_restart"]
    )

    generator.add_param(
        key="algo.name",
        name="algo_name",
        group=1,
        values=["flow_gnn"],
        value_names=["flow_gnn"]
    )

    generator.add_param(
        key="algo.inference_euler_steps",
        name="inference_euler_steps",
        group=1,
        values=[5],
        value_names=["5"]
    )

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
    print(f"2. Check the generated configs in the same directory as '{args.config}'.")
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