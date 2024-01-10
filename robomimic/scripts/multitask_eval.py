"""
Script for running evaluation on a single checkpoint or all checkpoints in a folder

Args:
    config_path (str): path to config file

    ckpt_path (str): path to checkpoint file

    style (str): how to select envs, currently only supports 'random' and 'all'

    num_envs (int): how many envs to select (only applicable if style is 'random')

    all: whether to run eval on all checkpoints in a folder (folder that contains ckpt_path)

Example usage:
    # evaluate a single checkpoint on 1 random env from config.train.data
    python multitask_eval.py --config_path /path/to/config.json --ckpt_path /path/to/checkpoint.pth --style random --num_envs 1

    # evaluate all checkpoints on the folder containig checkpoint.pth.  Eval run on all envs from config.train.data
    python multitask_eval.py --config_path /path/to/config.json --ckpt_path /path/to/checkpoint.pth --all

    # evaluate all checkpoints on the folder containig checkpoint.pth.  Eval run on 3 random envs from config.train.data
    python multitask_eval.py --config_path /path/to/config.json --ckpt_path /path/to/checkpoint.pth --style random --num_envs 3 --all

"""
import os
import json
from collections import OrderedDict
from robomimic.config import config_factory
import argparse
import re

import robomimic
import robomimic.utils.train_utils as TrainUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.torch_utils as TorchUtils


def select_envs(envs, style = 'random', num_envs = 1):
    """
        This function determines how the select envs, currently only random and all are supported
        Note: we are selecting from the envs specified in config.train.data
    """
    if style == 'random':
        import random
        env_keys = list(envs.keys())
        random.shuffle(env_keys)
        return {k: envs[k] for k in env_keys[:num_envs]}
    elif style == 'all':
        return envs
    else:
        raise NotImplementedError
    
def run_eval_single(config, ckpt_path, video_dir, log_dir, final_envs):
    """ 
        Internal helper function for running evaluation on a single checkpoint
        Would not recommend calling this function directly, specifying final_envs may be non-trivial
    """
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    rollout_model, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=ckpt_path, device=device, verbose=True)
    
    epoch = os.path.basename(ckpt_path).split("_")[2]
    epoch = re.sub(r'[^0-9]', '', epoch)
    epoch = int(epoch)
    num_episodes = config.experiment.rollout.n
    
    all_rollout_logs, video_paths = TrainUtils.rollout_with_stats(
        policy=rollout_model,
        envs=final_envs,
        horizon=config.experiment.rollout.horizon,
        use_goals=config.use_goals,
        num_episodes=num_episodes,
        render=False,
        video_dir=video_dir if config.experiment.render_video else None,
        epoch=epoch,
        video_skip=config.experiment.get("video_skip", 5),
        terminate_on_success=config.experiment.rollout.terminate_on_success,
    )

    avg_success_rate = 0
    for env_name in all_rollout_logs:
        rollout_logs = all_rollout_logs[env_name]
        print("\nEpoch {} Rollouts took {}s (avg) with results:".format(epoch, rollout_logs["time"]))
        print('Env: {}'.format(env_name))
        print('Success rate: {}'.format(rollout_logs["Success_Rate"]))
        avg_success_rate += rollout_logs["Success_Rate"]
    avg_success_rate /= len(all_rollout_logs)
    log_name = "epoch_{}_success_{}_logs.json".format(epoch, avg_success_rate)
    with open(os.path.join(log_dir, log_name), 'w') as file:
        json.dump(all_rollout_logs, file, indent=4)
    
    return avg_success_rate, epoch
    
def run_eval(config_path, ckpt_path, style = 'random', num_envs = 1, all = False):
    """
        Runs eval on a single checkpoint or all checkpoints in a folder
        Args:
            config_path: path to config file
            ckpt_path: path to checkpoint file
            style: how to select envs, currently only supports 'random' and 'all'
            num_envs: how many envs to select (only applicable if style is 'random')
            all: whether to run eval on all checkpoints in a folder (folder that contains ckpt_path)  
    """

    ext_cfg = json.load(open(config_path, 'r'))
    config = config_factory(ext_cfg["algo_name"])
    # update config with external json - this will throw errors if
    # the external config has keys not present in the base algo config
    with config.values_unlocked():
        config.update(ext_cfg)

    env_meta_list = []
    shape_meta_list = []

    log_dir, ckpt_dir, video_dir, vis_dir = TrainUtils.get_exp_dir(config)
    ObsUtils.initialize_obs_utils_with_config(config)

    for dataset_cfg in config.train.data:
            dataset_path = os.path.expanduser(dataset_cfg["path"])
            ds_format = config.train.data_format
            if not os.path.exists(dataset_path):
                raise Exception("Dataset at provided path {} not found!".format(dataset_path))

            # load basic metadata from training file
            print("\n============= Loaded Environment Metadata =============")
            env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=dataset_path, ds_format=ds_format)

            # populate language instruction for env in env_meta
            env_meta["lang"] = dataset_cfg.get("lang", "dummy")

            # update env meta if applicable
            from robomimic.utils.script_utils import deep_update
            deep_update(env_meta, config.experiment.env_meta_update_dict)
            env_meta_list.append(env_meta)

            shape_meta = FileUtils.get_shape_metadata_from_dataset(
                dataset_config=dataset_cfg,
                action_keys=config.train.action_keys,
                all_obs_keys=config.all_obs_keys,
                ds_format=ds_format,
                verbose=True
            )
            shape_meta_list.append(shape_meta)


    envs = OrderedDict()
    for (dataset_i, dataset_cfg) in enumerate(config.train.data):
        env_meta = env_meta_list[dataset_i]
        shape_meta = shape_meta_list[dataset_i]
        env_name = env_meta["env_name"]
        
        def create_env(env_i=0):
            env_kwargs = dict(
                env_meta=env_meta,
                env_name=env_name,
                render=False,
                render_offscreen=config.experiment.render_video,
                use_image_obs=shape_meta["use_images"],
                # seed=config.train.seed * 1000 + env_i # TODO: add seeding across environments
            )
            env = EnvUtils.create_env_from_metadata(**env_kwargs)
            # handle environment wrappers
            env = EnvUtils.wrap_env_from_config(env, config=config)  # apply environment warpper, if applicable

            return env

        if config.experiment.rollout.batched:
            from tianshou.env import SubprocVectorEnv
            env_fns = [lambda env_i=i: create_env(env_i) for i in range(config.experiment.rollout.num_batch_envs)]
            env = SubprocVectorEnv(env_fns)
            env_name = env.get_env_attr(key="name", id=0)[0]
        else:
            env = create_env()
            env_name = env.name
        
        env_key = os.path.splitext(os.path.basename(dataset_cfg['path']))[0]
        envs[env_key] = env
        print(env)

    final_envs = select_envs(envs, style = style, num_envs = num_envs)

    best_epoch = 0
    best_success_rate = 0
    # folder of checkpoints and run evaluation for each chekpoint in the folder:
    if all:
        ckpt_dir = os.path.dirname(ckpt_path)
        for ckpt_path in os.listdir(ckpt_dir):
            if not ckpt_path.endswith(".pth"):
                continue    
            print("Evaluating checkpoint: {}".format(ckpt_path))
            ckpt_path = os.path.join(ckpt_dir, ckpt_path)
            success, epoch = run_eval_single(config, ckpt_path, video_dir, log_dir, final_envs)
            if success > best_success_rate:
                best_epoch = epoch
                best_success_rate = success
    else:
        best_epoch, best_success_rate = run_eval_single(config, ckpt_path, video_dir, log_dir, final_envs)

    print("Best epoch: {}, success rate: {}".format(best_epoch, best_success_rate))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # path to config file
    parser.add_argument(
        "--config_path", 
        type=str, 
        required=True,
        help="path to config file, the one used for training (although you can use a different one)"
        )
    
    # path to checkpoint file
    parser.add_argument(
        "--ckpt_path", 
        type=str, 
        required=True,
        help="path to checkpoint file, the one to be evaluated. If all is specified, looks at all checkpoints in same folder as this file"
        )
    
    # method for env selection
    parser.add_argument(
        "--style", 
        type=str, 
        default='random',
        help="how to select envs, currently only supports 'random' and 'all'"
        )
    
    # num_envs for evaluation
    parser.add_argument(
        "--num_envs", 
        type=int, 
        default=1,
        help="how many envs to select (only applicable if style is 'random')"
        )
    
    # single checkpoint or all checkpoints in a folder
    parser.add_argument(
        "--all", 
        action='store_true',
        help="whether to run eval on all checkpoints in a folder (folder that contains ckpt_path)"
        )

    args = parser.parse_args()
    run_eval(args.config_path, args.ckpt_path, args.style, args.num_envs, args.all)