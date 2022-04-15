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
import robomimic.utils.macros as Macros
from robomimic.config import config_factory
from robomimic.scripts.train import train


def robosuite_hyperparameters(config):
    """
    Sets robosuite-specific hyperparameters.

    Args:
        config (Config): Config to modify

    Returns:
        Config: Modified config
    """
    ## save config - if and when to save checkpoints ##
    config.experiment.save.enabled = True                       # whether model saving should be enabled or disabled
    config.experiment.save.every_n_seconds = None               # save model every n seconds (set to None to disable)
    config.experiment.save.every_n_epochs = 50                  # save model every n epochs (set to None to disable)
    config.experiment.save.epochs = []                          # save model on these specific epochs
    config.experiment.save.on_best_validation = False           # save models that achieve best validation score
    config.experiment.save.on_best_rollout_return = False       # save models that achieve best rollout return
    config.experiment.save.on_best_rollout_success_rate = True  # save models that achieve best success rate

    # epoch definition - if not None, set an epoch to be this many gradient steps, else the full dataset size will be used
    config.experiment.epoch_every_n_steps = 100                 # each epoch is 100 gradient steps
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
    config.experiment.rollout.n = 50                            # number of rollouts per evaluation
    config.experiment.rollout.horizon = 400                     # maximum number of env steps per rollout
    config.experiment.rollout.rate = 50                         # do rollouts every @rate epochs
    config.experiment.rollout.warmstart = 0                     # number of epochs to wait before starting rollouts
    config.experiment.rollout.terminate_on_success = True       # end rollout early after task success

    ## dataset loader config ##

    # num workers for loading data - generally set to 0 for low-dim datasets, and 2 for image datasets
    config.train.num_data_workers = 0                           # assume low-dim dataset

    # One of ["all", "low_dim", or None]. Set to "all" to cache entire hdf5 in memory - this is
    # by far the fastest for data loading. Set to "low_dim" to cache all non-image data. Set
    # to None to use no caching - in this case, every batch sample is retrieved via file i/o.
    # You should almost never set this to None, even for large image datasets.
    config.train.hdf5_cache_mode = "all"

    config.train.hdf5_use_swmr = True                           # used for parallel data loading

    # if true, normalize observations at train and test time, using the global mean and standard deviation
    # of each observation in each dimension, computed across the training set. See SequenceDataset.normalize_obs
    # in utils/dataset.py for more information.
    config.train.hdf5_normalize_obs = False                     # no obs normalization

    # if provided, demonstrations are filtered by the list of demo keys under "mask/@hdf5_filter_key"
    config.train.hdf5_filter_key = None                         # by default, use no filter key

    # fetch sequences of length 10 from dataset for RNN training
    config.train.seq_length = 10

    # keys from hdf5 to load per demonstration, besides "obs" and "next_obs"
    config.train.dataset_keys = (
        "actions",
        "rewards",
        "dones",
    )

    # one of [None, "last"] - set to "last" to include goal observations in each batch
    config.train.goal_mode = None                               # no need for goal observations

    ## learning config ##
    config.train.cuda = True                                    # try to use GPU (if present) or not
    config.train.batch_size = 100                               # batch size
    config.train.num_epochs = 2000                              # number of training epochs
    config.train.seed = 1                                       # seed for training


    ### Observation Config ###
    config.observation.modalities.obs.low_dim = [               # specify low-dim observations for agent
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos",
        "object",
    ]
    config.observation.modalities.obs.rgb = []                # no image observations
    config.observation.modalities.goal.low_dim = []             # no low-dim goals
    config.observation.modalities.goal.rgb = []               # no image goals

    # observation encoder architecture - applies to all networks that take observation dicts as input

    config.observation.encoder.rgb.core_class = "VisualCore"
    config.observation.encoder.rgb.core_kwargs.feature_dimension = 64
    config.observation.encoder.rgb.core_kwargs.backbone_class = 'ResNet18Conv'                         # ResNet backbone for image observations (unused if no image observations)
    config.observation.encoder.rgb.core_kwargs.backbone_kwargs.pretrained = False                # kwargs for visual core
    config.observation.encoder.rgb.core_kwargs.backbone_kwargs.input_coord_conv = False
    config.observation.encoder.rgb.core_kwargs.pool_class = "SpatialSoftmax"                # Alternate options are "SpatialMeanPool" or None (no pooling)
    config.observation.encoder.rgb.core_kwargs.pool_kwargs.num_kp = 32                      # Default arguments for "SpatialSoftmax"
    config.observation.encoder.rgb.core_kwargs.pool_kwargs.learnable_temperature = False    # Default arguments for "SpatialSoftmax"
    config.observation.encoder.rgb.core_kwargs.pool_kwargs.temperature = 1.0                # Default arguments for "SpatialSoftmax"
    config.observation.encoder.rgb.core_kwargs.pool_kwargs.noise_std = 0.0                  # Default arguments for "SpatialSoftmax"

    # observation randomizer class - set to None to use no randomization, or 'CropRandomizer' to use crop randomization
    config.observation.encoder.rgb.obs_randomizer_class = None

    # kwargs for observation randomizers (for the CropRandomizer, this is size and number of crops)
    config.observation.encoder.rgb.obs_randomizer_kwargs.crop_height = 76
    config.observation.encoder.rgb.obs_randomizer_kwargs.crop_width = 76
    config.observation.encoder.rgb.obs_randomizer_kwargs.num_crops = 1
    config.observation.encoder.rgb.obs_randomizer_kwargs.pos_enc = False

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
    config.algo.actor_layer_dims = ()   # empty MLP - go from RNN layer directly to action output

    # stochastic GMM policy
    config.algo.gmm.enabled = True                      # enable GMM policy - policy outputs GMM action distribution
    config.algo.gmm.num_modes = 5                       # number of GMM modes
    config.algo.gmm.min_std = 0.0001                    # minimum std output from network
    config.algo.gmm.std_activation = "softplus"         # activation to use for std output from policy net
    config.algo.gmm.low_noise_eval = True               # low-std at test-time

    # rnn policy config
    config.algo.rnn.enabled = True      # enable RNN policy
    config.algo.rnn.horizon = 10        # unroll length for RNN - should usually match train.seq_length
    config.algo.rnn.hidden_dim = 400    # hidden dimension size
    config.algo.rnn.rnn_type = "LSTM"   # rnn type - one of "LSTM" or "GRU"
    config.algo.rnn.num_layers = 2      # number of RNN layers that are stacked
    config.algo.rnn.open_loop = False   # if True, action predictions are only based on a single observation (not sequence) + hidden state
    config.algo.rnn.kwargs.bidirectional = False          # rnn kwargs

    return config


def momart_hyperparameters(config):
    """
    Sets momart-specific hyperparameters.

    Args:
        config (Config): Config to modify

    Returns:
        Config: Modified config
    """
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
    config.experiment.rollout.rate = 3                          # do rollouts every @rate epochs
    config.experiment.rollout.warmstart = 0                     # number of epochs to wait before starting rollouts
    config.experiment.rollout.terminate_on_success = True       # end rollout early after task success

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
    config.observation.modalities.obs.rgb = [
        "rgb",
        "rgb_wrist",
    ]

    config.observation.modalities.obs.depth = [
        "depth",
        "depth_wrist",
    ]
    config.observation.modalities.obs.scan = [
        "scan",
    ]
    config.observation.modalities.goal.low_dim = []             # no low-dim goals
    config.observation.modalities.goal.rgb = []               # no rgb image goals

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

    return config


# Valid dataset types to use
DATASET_TYPES = {
    "robosuite": {
        "default_dataset_func": TestUtils.example_dataset_path,
        "hp": robosuite_hyperparameters,
    },
    "momart": {
        "default_dataset_func": TestUtils.example_momart_dataset_path,
        "hp": momart_hyperparameters,
    },
}


def get_config(dataset_type="robosuite", dataset_path=None, output_dir=None, debug=False):
    """
    Construct config for training.

    Args:
        dataset_type (str): Dataset type to use. Valid options are DATASET_TYPES. Default is "robosuite"
        dataset_path (str or None): path to hdf5 dataset. Pass None to use a small default dataset.
        output_dir (str): path to output folder, where logs, model checkpoints, and videos
            will be written. If it doesn't exist, the directory will be created. Pass
            None to use a default directory in /tmp.
        debug (bool): if True, shrink training and rollout times to test a full training
            run quickly.
    """
    assert dataset_type in DATASET_TYPES, \
        f"Invalid dataset type. Valid options are: {list(DATASET_TYPES.keys())}, got: {dataset_type}"

    # handle args
    if dataset_path is None:
        # small dataset with a handful of trajectories
        dataset_path = DATASET_TYPES[dataset_type]["default_dataset_func"]()

    if output_dir is None:
        # default output directory created in /tmp
        output_dir = TestUtils.temp_model_dir_path()

    # make default BC config
    config = config_factory(algo_name="bc")

    ### Experiment Config ###
    config.experiment.name = f"{dataset_type}_bc_rnn_example"   # name of experiment used to make log files
    config.experiment.validate = True                           # whether to do validation or not
    config.experiment.logging.terminal_output_to_txt = False    # whether to log stdout to txt file 
    config.experiment.logging.log_tb = True                     # enable tensorboard logging

    ### Train Config ###
    config.train.data = dataset_path                            # path to hdf5 dataset

    # Write all results to this directory. A new folder with the timestamp will be created
    # in this directory, and it will contain three subfolders - "log", "models", and "videos".
    # The "log" directory will contain tensorboard and stdout txt logs. The "models" directory
    # will contain saved model checkpoints. The "videos" directory contains evaluation rollout
    # videos.
    config.train.output_dir = output_dir                        # path to output folder

    # Load default hyperparameters based on dataset type
    config = DATASET_TYPES[dataset_type]["hp"](config)

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
        default=None,
        help="(optional) path to input hdf5 dataset to use in example script. If not provided, \
            a default hdf5 packaged with the repository will be used.",
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

    # type
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="robosuite",
        choices=list(DATASET_TYPES.keys()),
        help=f"Dataset type to use. This will determine the default hyperparameter settings to use for training."
             f"Valid options are: {list(DATASET_TYPES.keys())}. Default is robosuite."
    )

    args = parser.parse_args()

    # Turn debug mode on possibly
    if args.debug:
        Macros.DEBUG = True

    # config for training
    config = get_config(
        dataset_type=args.dataset_type,
        dataset_path=args.dataset,
        output_dir=args.output,
        debug=args.debug
    )

    # set torch device
    device = TorchUtils.get_torch_device(try_to_use_cuda=config.train.cuda)

    # run training
    train(config, device=device)
