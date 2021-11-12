"""
WARNING: This script is only for instructive purposes, to point out different portions
         of the config -- the preferred way to launch training runs is still with external
         jsons and scripts/train.py (and optionally using scripts/hyperparameter_helper.py
         to generate several config jsons by sweeping config settings). See the online
         documentation for more information about launching training.

Example script for training a BC-RNN agent by manually setting portions of the config in 
python code. 

To see a quick training run, use the following command:

    python train_bc_rnn.py --debug

To run a full length training run on your own dataset, use the following command:

    python train_bc_rnn.py --dataset /path/to/dataset.hdf5 --output /path/to/output_dir
"""
import argparse

import robomimic
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.test_utils as TestUtils
from robomimic.config import config_factory
from robomimic.scripts.train import train


def get_config(dataset_path=None, output_dir=None, debug=False):
    """
    Construct config for training.

    Args:
        dataset_path (str): path to hdf5 dataset. Pass None to use a small default dataset.
        output_dir (str): path to output folder, where logs, model checkpoints, and videos
            will be written. If it doesn't exist, the directory will be created. Pass
            None to use a default directory in /tmp.
        debug (bool): if True, shrink training and rollout times to test a full training
            run quickly.
    """
    # Get task name
    task = dataset_path.split(".")[-2].split("/")[-1]

    # handle args
    if dataset_path is None:
        # small dataset with a handful of trajectories
        dataset_path = TestUtils.example_dataset_path()

    if output_dir is None:
        # default output directory created in /tmp
        output_dir = TestUtils.temp_model_dir_path()

    # make default BC config
    config = config_factory(algo_name="bc")

    ### Experiment Config ###
    config.experiment.name = f"momart_{task}_bc_rnn"                   # name of experiment used to make log files
    config.experiment.validate = True                           # whether to do validation or not
    config.experiment.logging.terminal_output_to_txt = False    # whether to log stdout to txt file 
    config.experiment.logging.log_tb = True                     # enable tensorboard logging

    ## save config - if and when to save checkpoints ##
    config.experiment.save.enabled = True                       # whether model saving should be enabled or disabled
    config.experiment.save.every_n_seconds = None               # save model every n seconds (set to None to disable)
    config.experiment.save.every_n_epochs = 3                   # save model every n epochs (set to None to disable)
    config.experiment.save.epochs = []                          # save model on these specific epochs
    config.experiment.save.on_best_validation = True            # save models that achieve best validation score
    config.experiment.save.on_best_rollout_return = False       # save models that achieve best rollout return
    config.experiment.save.on_best_rollout_success_rate = True  # save models that achieve best success rate

    # epoch definition - if not None, set an epoch to be this many gradient steps, else the full dataset size will be used
    config.experiment.epoch_every_n_steps = None                # each epoch is 100 gradient steps
    config.experiment.validation_epoch_every_n_steps = 10       # each validation epoch is 10 gradient steps

    # envs to evaluate model on (assuming rollouts are enabled), to override the metadata stored in dataset
    config.experiment.env = None                                # no need to set this (unless you want to override)
    config.experiment.additional_envs = None                    # additional environments that should get evaluated

    ## rendering config ##
    config.experiment.render = False                            # render on-screen or not
    config.experiment.render_video = True                       # render evaluation rollouts to videos
    config.experiment.keep_all_videos = False                   # save all videos, instead of only saving those for saved model checkpoints
    config.experiment.video_skip = 5                            # render video frame every n environment steps during rollout

    ## evaluation rollout config ##
    config.experiment.rollout.enabled = True                    # enable evaluation rollouts
    config.experiment.rollout.n = 30                            # number of rollouts per evaluation
    config.experiment.rollout.horizon = 1500                    # maximum number of env steps per rollout
    config.experiment.rollout.rate = 100                        # do rollouts every @rate epochs
    config.experiment.rollout.warmstart = 0                     # number of epochs to wait before starting rollouts
    config.experiment.rollout.terminate_on_success = True       # end rollout early after task success


    ### Train Config ###
    config.train.data = dataset_path                            # path to hdf5 dataset

    # Write all results to this directory. A new folder with the timestamp will be created
    # in this directory, and it will contain three subfolders - "log", "models", and "videos".
    # The "log" directory will contain tensorboard and stdout txt logs. The "models" directory
    # will contain saved model checkpoints. The "videos" directory contains evaluation rollout
    # videos.
    config.train.output_dir = output_dir                        # path to output folder

    ## dataset loader config ##

    # num workers for loading data - generally set to 0 for low-dim datasets, and 2 for image datasets
    config.train.num_data_workers = 2                           # assume low-dim dataset

    # One of ["all", "low_dim", or None]. Set to "all" to cache entire hdf5 in memory - this is 
    # by far the fastest for data loading. Set to "low_dim" to cache all non-image data. Set
    # to None to use no caching - in this case, every batch sample is retrieved via file i/o.
    # You should almost never set this to None, even for large image datasets.
    config.train.hdf5_cache_mode = "low_dim"

    config.train.hdf5_use_swmr = True                           # used for parallel data loading

    # if true, normalize observations at train and test time, using the global mean and standard deviation
    # of each observation in each dimension, computed across the training set. See SequenceDataset.normalize_obs
    # in utils/dataset.py for more information.
    config.train.hdf5_normalize_obs = False                     # no obs normalization

    # if provided, demonstrations are filtered by the list of demo keys under "mask/@hdf5_filter_key"
    config.train.hdf5_filter_key = None                         # by default, use no filter key

    # fetch sequences of length 10 from dataset for RNN training
    config.train.seq_length = 50

    # keys from hdf5 to load per demonstration, besides "obs" and "next_obs"
    config.train.dataset_keys = (
        "actions", 
        "rewards", 
        "dones",
    )

    # one of [None, "last"] - set to "last" to include goal observations in each batch
    config.train.goal_mode = "last"                               # no need for goal observations

    ## learning config ##
    config.train.cuda = True                                    # try to use GPU (if present) or not
    config.train.batch_size = 4                               # batch size
    config.train.num_epochs = 31                              # number of training epochs
    config.train.seed = 1                                       # seed for training


    ### Observation Config ###
    config.observation.modalities.obs.low_dim = [               # specify low-dim observations for agent
        "proprio",
    ]
    config.observation.modalities.obs.image = [
        # "rgb",
        # "rgb_wrist",
    ]                # no image observations

    config.observation.modalities.obs.depth = [
        "depth",
        "depth_wrist",
    ]
    config.observation.modalities.obs.scan = [
        "scan",
    ]
    config.observation.modalities.goal.low_dim = []             # no low-dim goals
    config.observation.modalities.goal.image = []               # no image goals

    ### Algo Config ###

    # optimization parameters
    config.algo.optim_params.policy.learning_rate.initial = 1e-4        # policy learning rate
    config.algo.optim_params.policy.learning_rate.decay_factor = 0.1    # factor to decay LR by (if epoch schedule non-empty)
    config.algo.optim_params.policy.learning_rate.epoch_schedule = []   # epochs where LR decay occurs
    config.algo.optim_params.policy.regularization.L2 = 0.00            # L2 regularization strength

    # loss weights
    config.algo.loss.l2_weight = 1.0    # L2 loss weight
    config.algo.loss.l1_weight = 0.0    # L1 loss weight
    config.algo.loss.cos_weight = 0.0   # cosine loss weight

    # MLP network architecture (layers after observation encoder and RNN, if present)
    config.algo.actor_layer_dims = (300, 400)           # MLP layers between RNN layer and action output

    # stochastic GMM policy
    config.algo.gmm.enabled = True                      # enable GMM policy - policy outputs GMM action distribution
    config.algo.gmm.num_modes = 5                       # number of GMM modes
    config.algo.gmm.min_std = 0.01                      # minimum std output from network
    config.algo.gmm.std_activation = "softplus"         # activation to use for std output from policy net
    config.algo.gmm.low_noise_eval = True               # low-std at test-time 

    # rnn policy config
    config.algo.rnn.enabled = True      # enable RNN policy                                        
    config.algo.rnn.horizon = 50        # unroll length for RNN - should usually match train.seq_length
    config.algo.rnn.hidden_dim = 1200   # hidden dimension size
    config.algo.rnn.rnn_type = "LSTM"   # rnn type - one of "LSTM" or "GRU"
    config.algo.rnn.num_layers = 2      # number of RNN layers that are stacked
    config.algo.rnn.open_loop = False   # if True, action predictions are only based on a single observation (not sequence) + hidden state
    config.algo.rnn.kwargs.bidirectional = False          # rnn kwargs

    # maybe make training length small for a quick run
    if debug:

        # train and validate for 3 gradient steps per epoch, and 2 total epochs
        config.experiment.epoch_every_n_steps = 3
        config.experiment.validation_epoch_every_n_steps = 3
        config.train.num_epochs = 2

        # rollout and model saving every epoch, and make rollouts short
        config.experiment.save.every_n_epochs = 1
        config.experiment.rollout.rate = 1
        config.experiment.rollout.n = 2
        config.experiment.rollout.horizon = 10

    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Dataset path
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="absolute path to input hdf5 dataset.",
    )

    # Output dir
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="(optional) path to folder to use (or create) to output logs, model checkpoints, and rollout \
            videos. If not provided, a folder in /tmp will be used.",
    )

    # debug flag for quick training run
    parser.add_argument(
        "--debug",
        action='store_true',
        help="set this flag to run a quick training run for debugging purposes"
    )

    args = parser.parse_args()

    # Get task name
    task = args.dataset.split(".")[-2].split("/")[-1]

    # config for training
    config = get_config(dataset_path=args.dataset, output_dir=args.output, debug=args.debug)

    # Save config

    cfg_fname = f'momart_{task}_bc_rnn_config.json'
    config.dump(cfg_fname)

    # set torch device
    device = TorchUtils.get_torch_device(try_to_use_cuda=config.train.cuda)

    # run training
    train(config, device=device)
