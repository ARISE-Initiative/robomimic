#!/usr/bin/env python

"""
Simplified script to evaluate multiple Robomimic models in parallel across seeds.

Evaluates checkpoints found in specified directories using multiprocessing.
Removed semaphore logic and map_location argument for simplicity and compatibility.
If CUDA errors occur, try reducing --n_workers.

Args:
    model_dir (str): Path(s) to directory/directories containing model checkpoints
                     (.pth files). Searched recursively. (Required)
    n_rollouts (int): Number of evaluation rollouts per model per seed. (Default: 10)
    seeds (int or list[int]): Evaluation seeds. Provide one integer N for N random
                               seeds, or multiple integers for specific seeds. (Required)
    horizon (int): If provided, override maximum rollout horizon from checkpoint config.
    env (str): If provided, override environment name from checkpoint config.
    n_workers (int): Number of parallel CPU workers. (Default: 0 = auto-detect, min(cpu_count, 4))
                     Reduce if encountering CUDA errors.
    output_plot (str): Path to save the output plot (e.g., 'results.png').
    output_json (str): Path to save detailed evaluation results as JSON.

Example usage:

    # Evaluate all models in 'my_models', 10 rollouts/seed, 3 random seeds,
    # using 4 CPU workers, save plot & json.
    python evaluate_all_simple.py --model_dir my_models --n_rollouts 10 \\
        --seeds 3 --n_workers 4 \\
        --output_plot comparison.png --output_json results.json

    # Evaluate with specific seeds 0, 10, 20 using 2 workers.
    python evaluate_all_simple.py --model_dir my_models --n_rollouts 20 \\
        --seeds 0 10 20 --n_workers 2
"""

import argparse
import json
import os
import glob
import numpy as np
import multiprocessing as mp
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
import torch
import traceback
from copy import deepcopy

# Robomimic Imports
try:
    import robomimic
    import robomimic.utils.file_utils as FileUtils
    import robomimic.utils.torch_utils as TorchUtils
    import robomimic.utils.tensor_utils as TensorUtils
    import robomimic.utils.obs_utils as ObsUtils
    from robomimic.envs.env_base import EnvBase
    from robomimic.envs.wrappers import EnvWrapper
    from robomimic.algo import RolloutPolicy
except ImportError as e:
    print(f"Error importing robomimic: {e}")
    print("Please ensure robomimic is installed (`pip install robomimic`) and accessible.")
    exit(1)

# --- Rollout Function (Copied & Simplified) ---
def rollout(policy, env, horizon):
    """Helper function to carry out a single rollout and return stats."""
    assert isinstance(env, EnvBase) or isinstance(env, EnvWrapper)
    assert isinstance(policy, RolloutPolicy)

    policy.start_episode()
    try:
        obs = env.reset()
        state_dict = env.get_state()
        if hasattr(env, 'reset_to'): obs = env.reset_to(state_dict)
    except Exception as e:
         print(f"\nError during env reset for {env.name}: {e}")
         return dict(Return=0.0, Horizon=0, Success_Rate=0.0), {}

    total_reward = 0.
    success = False
    step_i = 0

    try:
        for step_i in range(horizon):
            with torch.no_grad(): act = policy(ob=obs)
            next_obs, r, done, info = env.step(act)

            success_info = env.is_success()
            if isinstance(success_info, dict) and 'task' in success_info: success = success_info['task']
            elif isinstance(success_info, bool): success = success_info
            else: success = False

            total_reward += r
            if done or success: break
            obs = deepcopy(next_obs)

    except env.rollout_exceptions as e:
        print(f"\nWarning: Rollout exception for {env.name}: {e}")
    except Exception as e:
        print(f"\nError during rollout step {step_i} for {env.name}: {e}\n{traceback.format_exc()}")
        success = False
        actual_horizon = step_i + 1
        stats = dict(Return=total_reward, Horizon=actual_horizon, Success_Rate=float(success))
        return stats, {}

    actual_horizon = step_i + 1
    stats = dict(Return=total_reward, Horizon=actual_horizon, Success_Rate=float(success))
    return stats, {}


# --- Top-Level Helper Functions ---

def find_checkpoints(model_dirs):
    """Find all model checkpoint files (.pth) in the given directories."""
    checkpoint_files = []
    print("\nSearching for checkpoints in:")
    for model_dir in model_dirs:
        print(f" - {model_dir}")
        norm_path = os.path.normpath(model_dir)
        found_in_dir = glob.glob(os.path.join(glob.escape(norm_path), "**/*.pth"), recursive=True)
        if not found_in_dir: print(f"   Warning: No .pth files found in {model_dir}")
        checkpoint_files.extend(found_in_dir)
    checkpoint_files.sort()
    return checkpoint_files

# Core worker function: Evaluates one checkpoint for one seed
def evaluate_checkpoint(args_tuple):
    """
    Worker function: Evaluates a single checkpoint for a given seed.
    Accepts a tuple of arguments for use with pool.map.
    """
    checkpoint_path, seed, n_rollouts, horizon, env_name = args_tuple # Unpack tuple

    result = {
        "checkpoint": checkpoint_path,
        "seed": seed,
        "success_rate_avg": 0.0,
        "success_rate_std": 0.0,
        "success_rates": [],
        "rollout_returns_avg": 0.0,
        "rollout_horizons_avg": 0.0,
        "error": None
    }
    policy = None
    env = None

    try:
        # Set seed for this specific worker/task
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Determine device (CUDA if available, else CPU)
        device = TorchUtils.get_torch_device(try_to_use_cuda=True)

        # Restore Policy & Env
        # *** Removed map_location=device from this call ***
        policy, ckpt_dict = FileUtils.policy_from_checkpoint(
            ckpt_path=checkpoint_path,
            device=device,
            verbose=False
        )
        policy.eval()

        eval_horizon = horizon
        if eval_horizon is None:
            config, _ = FileUtils.config_from_checkpoint(ckpt_dict=ckpt_dict)
            eval_horizon = config.experiment.rollout.horizon

        env, _ = FileUtils.env_from_checkpoint(
            ckpt_dict=ckpt_dict, env_name=env_name, render=False, render_offscreen=False, verbose=False
        )

        # Run N Rollouts
        rollout_successes = []
        rollout_returns = []
        rollout_horizons = []
        for _ in range(n_rollouts):
            stats, _ = rollout(policy=policy, env=env, horizon=eval_horizon)
            rollout_successes.append(stats["Success_Rate"])
            rollout_returns.append(stats["Return"])
            rollout_horizons.append(stats["Horizon"])

        # Aggregate Statistics
        if rollout_successes:
            result["success_rates"] = rollout_successes
            result["success_rate_avg"] = float(np.mean(rollout_successes))
            result["success_rate_std"] = float(np.std(rollout_successes))
            result["rollout_returns_avg"] = float(np.mean(rollout_returns))
            result["rollout_horizons_avg"] = float(np.mean(rollout_horizons))
        else:
             result["success_rates"] = []

    except Exception as e:
        result["error"] = f"Chkpt: {os.path.basename(checkpoint_path)}, Seed: {seed}\nError: {e}\nTraceback:\n{traceback.format_exc()}"
        # print(f"\n!!! Worker Error: {result['error']}") # Uncomment for immediate debug print

    finally:
        # Cleanup
        if hasattr(env, 'close'):
            try: env.close()
            except Exception as close_e: print(f"\nWarning: Error closing env for {os.path.basename(checkpoint_path)} seed {seed}: {close_e}")
        del policy
        del env
        if 'device' in locals() and device.type == 'cuda':
            try: torch.cuda.empty_cache()
            except Exception: pass

    return result


# Parallel Evaluation Orchestrator (Simplified)
def evaluate_checkpoints_parallel(checkpoint_paths, seeds, n_rollouts, horizon, env_name, n_workers):
    """Runs evaluation for all checkpoints and seeds in parallel."""
    eval_args = []
    for checkpoint_path in checkpoint_paths:
        for seed in seeds:
            eval_args.append((checkpoint_path, seed, n_rollouts, horizon, env_name))

    results = []
    print(f"\nLaunching {len(eval_args)} evaluation tasks using {n_workers} workers...")
    with mp.Pool(processes=n_workers) as pool:
        for result in tqdm(
            pool.imap_unordered(evaluate_checkpoint, eval_args),
            total=len(eval_args),
            desc="Evaluating models"
        ):
            results.append(result)
    return results


# Plotting Function
def plot_results(results, n_rollouts_per_seed, output_path=None):
    """Plots success rates for each checkpoint, aggregating across seeds."""
    checkpoint_results = defaultdict(list)
    successful_results = [r for r in results if r["error"] is None]

    if not successful_results:
        print("\nNo successful evaluations to plot.")
        return

    for result in successful_results:
        checkpoint_results[result["checkpoint"]].append(result)

    if not checkpoint_results:
        print("\nNo valid checkpoint results found after filtering errors.")
        return

    model_plot_names = []
    avg_success_rates_per_model = []
    std_success_rates_across_seeds = []
    all_rollout_rates_per_model = []

    all_checkpoint_paths = list(checkpoint_results.keys())
    try:
        common_prefix = os.path.commonpath([os.path.dirname(p) for p in all_checkpoint_paths])
        if os.path.basename(common_prefix) == 'models': common_prefix = os.path.dirname(common_prefix)
        common_prefix += os.sep
    except ValueError: common_prefix = ""

    num_seeds_found = 0
    for cp, seed_results_list in checkpoint_results.items():
        if not seed_results_list: continue
        num_seeds_found = len(seed_results_list)

        try:
            relative_path = cp[len(common_prefix):] if common_prefix and cp.startswith(common_prefix) else cp
            parts = relative_path.split(os.sep)
            model_name_label = f"{parts[-3]}/{parts[-1].replace('.pth', '')}" if len(parts) >= 3 and parts[-2] == 'models' else relative_path.replace('.pth','')
        except Exception: model_name_label = os.path.basename(cp).replace('.pth', '')
        model_plot_names.append(model_name_label)

        seed_avg_success_rates = [res["success_rate_avg"] for res in seed_results_list]
        avg_success_rates_per_model.append(np.mean(seed_avg_success_rates) if seed_avg_success_rates else 0.0)
        std_success_rates_across_seeds.append(np.std(seed_avg_success_rates) if seed_avg_success_rates else 0.0)

        rollout_rates_for_this_model = []
        for res in seed_results_list: rollout_rates_for_this_model.extend(res.get("success_rates", []))
        all_rollout_rates_per_model.append(rollout_rates_for_this_model)

    if not model_plot_names:
        print("\nNo models available for plotting after processing results.")
        return

    sorted_indices = None
    try:
        sorted_indices = np.argsort(avg_success_rates_per_model)[::-1]
        model_plot_names = [model_plot_names[i] for i in sorted_indices]
        avg_success_rates_per_model = [avg_success_rates_per_model[i] for i in sorted_indices]
        std_success_rates_across_seeds = [std_success_rates_across_seeds[i] for i in sorted_indices]
        all_rollout_rates_per_model = [all_rollout_rates_per_model[i] for i in sorted_indices]
    except (IndexError, TypeError):
        print("\nWarning: Could not sort results.")

    plt.figure(figsize=(max(12, len(model_plot_names) * 0.4), 8))
    x = np.arange(len(model_plot_names))

    bars = plt.bar(x, avg_success_rates_per_model, yerr=std_success_rates_across_seeds,
                   align='center', alpha=0.6, capsize=4,
                   label=f'Avg Success Rate ({n_rollouts_per_seed} rollouts/seed)')

    for i, rollout_rates in enumerate(all_rollout_rates_per_model):
        if rollout_rates:
            jitter = np.random.normal(0, 0.08, size=len(rollout_rates))
            plt.scatter(x=[i + j for j in jitter], y=rollout_rates,
                        color='black', alpha=0.25, s=15, zorder=10,
                        label='Individual Rollout Success' if i == 0 else "")

    plt.xlabel('Model (Experiment / Checkpoint)')
    plt.ylabel('Success Rate')
    plt.title(f'Model Success Rates (Sorted | {num_seeds_found} Seeds | Error Bars = Std Dev across Seeds)')
    plt.xticks(x, model_plot_names, rotation=70, ha='right', fontsize=max(6, 10 - len(model_plot_names)//10))
    plt.ylim(-0.05, 1.1)
    plt.grid(axis='y', linestyle=':', alpha=0.7)
    plt.legend(loc='upper right')
    plt.tight_layout(pad=1.5)

    for i, bar in enumerate(bars):
        plt.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + std_success_rates_across_seeds[i] + 0.02,
                 f'{avg_success_rates_per_model[i]:.3f}', ha='center', va='bottom',
                 fontsize=max(5, 8 - len(model_plot_names)//15),
                 bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.5, ec='none'))

    if output_path:
        try:
            plt.savefig(output_path, bbox_inches='tight', dpi=150)
            print(f"\nPlot saved to {output_path}")
        except Exception as plot_e: print(f"\nError saving plot: {plot_e}")

    try: plt.show()
    except Exception as show_e: print(f"\nCould not display plot: {show_e}")

    # Print Sorted Results
    print("\n--- Sorted Model Success Rates (Avg ± Std across Seeds) ---")
    max_name_len = max(len(name) for name in model_plot_names) if model_plot_names else 0

    avg_returns_per_model_sorted = []
    avg_horizons_per_model_sorted = []
    sorted_checkpoint_paths = all_checkpoint_paths # Default if sorting failed
    if sorted_indices is not None:
         # Use sorted_indices if sorting was successful
         sorted_checkpoint_paths = [all_checkpoint_paths[i] for i in sorted_indices]

    for cp in sorted_checkpoint_paths:
        seed_results_list = checkpoint_results[cp]
        if not seed_results_list:
            avg_returns_per_model_sorted.append(np.nan)
            avg_horizons_per_model_sorted.append(np.nan)
            continue
        avg_returns_per_model_sorted.append(np.mean([res.get("rollout_returns_avg", np.nan) for res in seed_results_list]))
        avg_horizons_per_model_sorted.append(np.mean([res.get("rollout_horizons_avg", np.nan) for res in seed_results_list]))

    print(f"{'Model':<{max_name_len}} | {'Success Rate':^18} | {'Avg Return':^12} | {'Avg Horizon':^12}")
    print("-" * (max_name_len + 1 + 18 + 3 + 12 + 3 + 12))
    for i, name in enumerate(model_plot_names):
        success_str = f"{avg_success_rates_per_model[i]:.4f} ± {std_success_rates_across_seeds[i]:.4f}"
        return_str = f"{avg_returns_per_model_sorted[i]:.1f}" if i < len(avg_returns_per_model_sorted) and not np.isnan(avg_returns_per_model_sorted[i]) else "N/A"
        horizon_str = f"{avg_horizons_per_model_sorted[i]:.1f}" if i < len(avg_horizons_per_model_sorted) and not np.isnan(avg_horizons_per_model_sorted[i]) else "N/A"
        print(f"{name:<{max_name_len}} | {success_str:^18} | {return_str:^12} | {horizon_str:^12}")


# --- Main Execution Logic ---
def main(args):
    try:
        mp.set_start_method('spawn', force=True)
        print("Set multiprocessing start method to 'spawn'.")
    except RuntimeError:
        print("Multiprocessing context already started. Using existing context.")

    checkpoint_paths = find_checkpoints(args.model_dir)
    if not checkpoint_paths:
        print(f"Error: No checkpoint (.pth) files found in: {args.model_dir}")
        return
    print(f"\nFound {len(checkpoint_paths)} checkpoint files.")

    if len(args.seeds) == 1:
        num_seeds = args.seeds[0]
        if num_seeds <= 0: print("Error: Number of seeds must be positive."); return
        master_rng = np.random.RandomState(0)
        seeds = master_rng.randint(0, 10000, size=num_seeds).tolist()
        print(f"\nGenerating {num_seeds} random evaluation seeds...")
    else:
        seeds = args.seeds
        print(f"\nUsing specific evaluation seeds: {seeds}")
    print(f"Seeds to use: {seeds}")

    n_workers = args.n_workers
    max_cpus = mp.cpu_count()
    if n_workers <= 0:
        n_workers = max(1, min(max_cpus, 4))
        print(f"Auto-detected {max_cpus} CPUs. Using {n_workers} parallel workers (default simple).")
    elif n_workers > max_cpus:
        print(f"Warning: Requested {n_workers} workers > {max_cpus} CPUs. Using {max_cpus}.")
        n_workers = max_cpus
    else:
         print(f"Using {n_workers} parallel workers.")

    start_time = time.time()
    results = evaluate_checkpoints_parallel(
        checkpoint_paths=checkpoint_paths,
        seeds=seeds,
        n_rollouts=args.n_rollouts,
        horizon=args.horizon,
        env_name=args.env,
        n_workers=n_workers,
    )
    end_time = time.time()
    print(f"\nTotal evaluation time: {end_time - start_time:.2f} seconds")

    errors = [r for r in results if r["error"] is not None]
    if errors:
        print(f"\n--- Encountered {len(errors)} Errors During Evaluation ---")
        for i, err in enumerate(errors[:min(len(errors), 5)]):
             model_name = os.path.basename(err['checkpoint'])
             print(f"\nError {i+1}/{len(errors)}: Model={model_name}, Seed={err['seed']}")
             error_lines = err['error'].splitlines()
             print('\n'.join(error_lines[:10]))
             if len(error_lines) > 10: print("...")
        if len(errors) > 5: print(f"... and {len(errors) - 5} more errors (see JSON output).")
        print("-" * 50)

    valid_results = [r for r in results if r["error"] is None]
    if not valid_results:
        print("\nNo valid results obtained. Cannot plot or save.")
        return

    plot_results(valid_results, n_rollouts_per_seed=args.n_rollouts, output_path=args.output_plot)

    if args.output_json:
        serializable_results = []
        for r in results:
            temp_r = r.copy()
            for key, value in temp_r.items():
                if isinstance(value, (np.float32, np.float64)): temp_r[key] = float(value)
                elif isinstance(value, list) and value and isinstance(value[0], (np.float32, np.float64)): temp_r[key] = [float(v) for v in value]
                elif isinstance(value, np.ndarray): temp_r[key] = value.tolist()
            serializable_results.append(temp_r)
        try:
            with open(args.output_json, 'w') as f:
                json.dump(serializable_results, f, indent=4)
            print(f"Full evaluation results saved to {args.output_json}")
        except Exception as json_e: print(f"Error saving results to JSON: {json_e}")


# --- Argument Parser and Main Guard ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate multiple Robomimic models in parallel across seeds (Simplified).")

    parser.add_argument("--model_dir", type=str, required=True, nargs='+', help="Path(s) to directory/directories containing checkpoints (.pth).")
    parser.add_argument("--seeds", type=int, nargs='+', required=True, help="Evaluation seeds: 1 int N for N random seeds, or list of specific seeds.")
    parser.add_argument("--n_rollouts", type=int, default=10, help="Number of rollouts per model per seed (default: 10)")
    parser.add_argument("--horizon", type=int, default=None, help="(Optional) Override rollout horizon.")
    parser.add_argument("--env", type=str, default=None, help="(Optional) Override environment name.")
    parser.add_argument("--n_workers", type=int, default=0, help="Number of parallel workers (default: 0 = auto-detect, min(cpu_count, 4)). Reduce if CUDA errors occur.")
    parser.add_argument("--output_plot", type=str, default=None, help="(Optional) Path to save success rate plot.")
    parser.add_argument("--output_json", type=str, default=None, help="(Optional) Path to save detailed results as JSON.")

    args = parser.parse_args()

    if args.n_rollouts <= 0: parser.error("--n_rollouts must be positive.")

    main(args)