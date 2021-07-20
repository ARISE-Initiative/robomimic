"""
A convenience script to report the average return of the last 10 rollout evaluations.
This is useful for collecting D4RL results, to be consistent with how the TD3_BC paper
(https://arxiv.org/abs/2106.06860) reported their results.
"""
import os
import json
import argparse
import numpy as np
import gym
import d4rl
import tensorflow as tf


def read_tb_log(tb_file, env_name):
    """
    Helper function to extract rollout results from a tensorboard log.
    """
    results = dict()
    verify_env_name = None
    for e in tf.compat.v1.train.summary_iterator(tb_file):
        epoch = e.step
        for v in e.summary.value:
            tag = v.tag
            exclude_suffixes = ["mean", "std", "min", "max"]
            if v.tag.startswith("Rollout/Return/") and not any([v.tag.endswith(suff) for suff in exclude_suffixes]):
                results[e.step] = v.simple_value
                verify_env_name = v.tag.split("/")[-1]
                assert env_name == verify_env_name
    sorted_epochs = sorted([x for x in list(results.keys())])
    assert len(sorted_epochs) >= 10, "must have at least 10 evaluations but got {}".format(len(sorted_epochs))
    assert sorted_epochs[-1] == 200, "last epoch must be 200"
    score = np.mean([results[e] for e in sorted_epochs[-10:]])
    env = gym.make(env_name)
    normalized_score = env.get_normalized_score(score)
    return score, normalized_score


def read_all_tb_logs(base_dir):
    """
    Reads all tensorboard logs in all subdirectories. Assumes the following directory structure:

    [env_name]
        trained_models
            [exp_name]
                [timestamp_dir]
                    logs
                        tb
                            [tb_file]
                    models
                    videos
    """
    results = {}
    for env_name in sorted(os.listdir(base_dir)):
        env_dir = os.path.join(base_dir, env_name)
        assert len(os.listdir(env_dir)) == 1
        trained_models_dir = os.path.join(env_dir, os.listdir(env_dir)[0])
        assert len(os.listdir(trained_models_dir)) == 1
        exp_dir = os.path.join(trained_models_dir, os.listdir(trained_models_dir)[0])
        assert len(os.listdir(exp_dir)) == 1
        timestamp_dir = os.path.join(exp_dir, os.listdir(exp_dir)[0])
        tb_dir = os.path.join(timestamp_dir, "logs", "tb")
        assert len(os.listdir(tb_dir)) == 1
        tb_file = os.path.join(tb_dir, os.listdir(tb_dir)[0])
        score, normalized_score = read_tb_log(tb_file=tb_file, env_name=env_name)
        results[env_name] = dict(score=score, normalized_score=normalized_score)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir",
        type=str,
        help="path to top-level directory",
    )

    args = parser.parse_args()
    results = read_all_tb_logs(base_dir=args.dir)
    print(json.dumps(results, indent=4))

