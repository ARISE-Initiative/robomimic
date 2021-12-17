"""
The base config class that is used for all algorithm configs in this repository.
Subclasses get registered into a global dictionary, making it easy to instantiate
the correct config class given the algorithm name.
"""

import six # preserve metaclass compatibility between python 2 and 3
from copy import deepcopy

import robomimic
from robomimic.config.config import Config

# global dictionary for remembering name - class mappings
REGISTERED_CONFIGS = {}


def get_all_registered_configs():
    """
    Give access to dictionary of all registered configs for external use.
    """
    return deepcopy(REGISTERED_CONFIGS)


def config_factory(algo_name, dic=None):
    """
    Creates an instance of a config from the algo name. Optionally pass
    a dictionary to instantiate the config from the dictionary.
    """
    if algo_name not in REGISTERED_CONFIGS:
        raise Exception("Config for algo name {} not found. Make sure it is a registered config among: {}".format(
            algo_name, ', '.join(REGISTERED_CONFIGS)))
    return REGISTERED_CONFIGS[algo_name](dict_to_load=dic)


class ConfigMeta(type):
    """
    Define a metaclass for constructing a config class.
    It registers configs into the global registry.
    """
    def __new__(meta, name, bases, class_dict):
        cls = super(ConfigMeta, meta).__new__(meta, name, bases, class_dict)
        if cls.__name__ != "BaseConfig":
            REGISTERED_CONFIGS[cls.ALGO_NAME] = cls
        return cls


@six.add_metaclass(ConfigMeta)
class BaseConfig(Config):
    def __init__(self, dict_to_load=None):
        if dict_to_load is not None:
            super(BaseConfig, self).__init__(dict_to_load)
            return

        super(BaseConfig, self).__init__()

        # store algo name class property in the config (must be implemented by subclasses)
        self.algo_name = type(self).ALGO_NAME

        self.experiment_config()
        self.train_config()
        self.algo_config()
        self.observation_config()

        # After Config init, new keys cannot be added to the config, except under nested
        # attributes that have called @do_not_lock_keys
        self.lock_keys()

    @property
    @classmethod
    def ALGO_NAME(cls):
        # must be specified by subclasses
        raise NotImplementedError

    def experiment_config(self):
        """
        This function populates the `config.experiment` attribute of the config, 
        which has several experiment settings such as the name of the training run, 
        whether to do logging, whether to save models (and how often), whether to render 
        videos, and whether to do rollouts (and how often). This class has a default 
        implementation that usually doesn't need to be overriden.
        """

        self.experiment.name = "test"                               # name of experiment used to make log files
        self.experiment.validate = True                             # whether to do validation or not
        self.experiment.logging.terminal_output_to_txt = True       # whether to log stdout to txt file 
        self.experiment.logging.log_tb = True                       # enable tensorboard logging


        ## save config - if and when to save model checkpoints ##
        self.experiment.save.enabled = True                         # whether model saving should be enabled or disabled
        self.experiment.save.every_n_seconds = None                 # save model every n seconds (set to None to disable)
        self.experiment.save.every_n_epochs = 50                    # save model every n epochs (set to None to disable)
        self.experiment.save.epochs = []                            # save model on these specific epochs
        self.experiment.save.on_best_validation = False             # save models that achieve best validation score
        self.experiment.save.on_best_rollout_return = False         # save models that achieve best rollout return
        self.experiment.save.on_best_rollout_success_rate = True    # save models that achieve best success rate

        # epoch definitions - if not None, set an epoch to be this many gradient steps, else the full dataset size will be used
        self.experiment.epoch_every_n_steps = 100                   # number of gradient steps in train epoch (None for full dataset pass)
        self.experiment.validation_epoch_every_n_steps = 10         # number of gradient steps in valid epoch (None for full dataset pass)

        # envs to evaluate model on (assuming rollouts are enabled), to override the metadata stored in dataset
        self.experiment.env = None                                  # no need to set this (unless you want to override)
        self.experiment.additional_envs = None                      # additional environments that should get evaluated


        ## rendering config ##
        self.experiment.render = False                              # render on-screen or not
        self.experiment.render_video = True                         # render evaluation rollouts to videos
        self.experiment.keep_all_videos = False                     # save all videos, instead of only saving those for saved model checkpoints
        self.experiment.video_skip = 5                              # render video frame every n environment steps during rollout


        ## evaluation rollout config ##
        self.experiment.rollout.enabled = True                      # enable evaluation rollouts
        self.experiment.rollout.n = 50                              # number of rollouts per evaluation
        self.experiment.rollout.horizon = 400                       # maximum number of env steps per rollout
        self.experiment.rollout.rate = 50                           # do rollouts every @rate epochs
        self.experiment.rollout.warmstart = 0                       # number of epochs to wait before starting rollouts
        self.experiment.rollout.terminate_on_success = True         # end rollout early after task success

    def train_config(self):
        """
        This function populates the `config.train` attribute of the config, which 
        has several settings related to the training process, such as the dataset 
        to use for training, and how the data loader should load the data. This 
        class has a default implementation that usually doesn't need to be overriden.
        """

        # Path to hdf5 dataset to use for training
        self.train.data = None                                      

        # Write all results to this directory. A new folder with the timestamp will be created
        # in this directory, and it will contain three subfolders - "log", "models", and "videos".
        # The "log" directory will contain tensorboard and stdout txt logs. The "models" directory
        # will contain saved model checkpoints. The "videos" directory contains evaluation rollout
        # videos.
        self.train.output_dir = "../{}_trained_models".format(self.algo_name)


        ## dataset loader config ##

        # num workers for loading data - generally set to 0 for low-dim datasets, and 2 for image datasets
        self.train.num_data_workers = 0  

        # One of ["all", "low_dim", or None]. Set to "all" to cache entire hdf5 in memory - this is 
        # by far the fastest for data loading. Set to "low_dim" to cache all non-image data. Set
        # to None to use no caching - in this case, every batch sample is retrieved via file i/o.
        # You should almost never set this to None, even for large image datasets.
        self.train.hdf5_cache_mode = "all"

        # used for parallel data loading
        self.train.hdf5_use_swmr = True

        # if true, normalize observations at train and test time, using the global mean and standard deviation
        # of each observation in each dimension, computed across the training set. See SequenceDataset.normalize_obs
        # in utils/dataset.py for more information.
        self.train.hdf5_normalize_obs = False

        # if provided, use the list of demo keys under the hdf5 group "mask/@hdf5_filter_key" for training, instead 
        # of the full dataset. This provides a convenient way to train on only a subset of the trajectories in a dataset.
        self.train.hdf5_filter_key = None

        # length of experience sequence to fetch from the dataset
        self.train.seq_length = 1

        # keys from hdf5 to load into each batch, besides "obs" and "next_obs". If algorithms
        # require additional keys from each trajectory in the hdf5, they should be specified here.
        self.train.dataset_keys = (
            "actions", 
            "rewards", 
            "dones",
        )

        # one of [None, "last"] - set to "last" to include goal observations in each batch
        self.train.goal_mode = None


        ## learning config ##
        self.train.cuda = True          # use GPU or not
        self.train.batch_size = 100     # batch size
        self.train.num_epochs = 2000    # number of training epochs
        self.train.seed = 1             # seed for training (for reproducibility)

    def algo_config(self):
        """
        This function populates the `config.algo` attribute of the config, and is given to the 
        `Algo` subclass (see `algo/algo.py`) for each algorithm through the `algo_config` 
        argument to the constructor. Any parameter that an algorithm needs to determine its 
        training and test-time behavior should be populated here. This function should be 
        implemented by every subclass.
        """
        pass

    def observation_config(self):
        """
        This function populates the `config.observation` attribute of the config, and is given 
        to the `Algo` subclass (see `algo/algo.py`) for each algorithm through the `obs_config` 
        argument to the constructor. This portion of the config is used to specify what 
        observation modalities should be used by the networks for training, and how the 
        observation modalities should be encoded by the networks. While this class has a 
        default implementation that usually doesn't need to be overriden, certain algorithm 
        configs may choose to, in order to have seperate configs for different networks 
        in the algorithm. 
        """

        # observation modalities
        self.observation.modalities.obs.low_dim = [             # specify low-dim observations for agent
            "robot0_eef_pos", 
            "robot0_eef_quat", 
            "robot0_gripper_qpos", 
            "object",
        ]
        self.observation.modalities.obs.rgb = []              # specify rgb image observations for agent
        self.observation.modalities.obs.depth = []
        self.observation.modalities.obs.scan = []
        self.observation.modalities.goal.low_dim = []           # specify low-dim goal observations to condition agent on
        self.observation.modalities.goal.rgb = []             # specify rgb image goal observations to condition agent on
        self.observation.modalities.goal.depth = []
        self.observation.modalities.goal.scan = []
        self.observation.modalities.obs.do_not_lock_keys()
        self.observation.modalities.goal.do_not_lock_keys()

        # observation encoder architectures (per obs modality)
        # This applies to all networks that take observation dicts as input

        # =============== Low Dim default encoder (no encoder) ===============
        self.observation.encoder.low_dim.core_class = None
        self.observation.encoder.low_dim.core_kwargs = Config()                 # No kwargs by default
        self.observation.encoder.low_dim.core_kwargs.do_not_lock_keys()

        # Low Dim: Obs Randomizer settings
        self.observation.encoder.low_dim.obs_randomizer_class = None
        self.observation.encoder.low_dim.obs_randomizer_kwargs = Config()       # No kwargs by default
        self.observation.encoder.low_dim.obs_randomizer_kwargs.do_not_lock_keys()

        # =============== RGB default encoder (ResNet backbone + linear layer output) ===============
        self.observation.encoder.rgb.core_class = "VisualCore"
        self.observation.encoder.rgb.core_kwargs.feature_dimension = 64
        self.observation.encoder.rgb.core_kwargs.flatten = True
        self.observation.encoder.rgb.core_kwargs.backbone_class = "ResNet18Conv"
        self.observation.encoder.rgb.core_kwargs.backbone_kwargs.pretrained = False
        self.observation.encoder.rgb.core_kwargs.backbone_kwargs.input_coord_conv = False
        self.observation.encoder.rgb.core_kwargs.backbone_kwargs.do_not_lock_keys()
        self.observation.encoder.rgb.core_kwargs.pool_class = "SpatialSoftmax"                # Alternate options are "SpatialMeanPool" or None (no pooling)
        self.observation.encoder.rgb.core_kwargs.pool_kwargs.num_kp = 32                      # Default arguments for "SpatialSoftmax"
        self.observation.encoder.rgb.core_kwargs.pool_kwargs.learnable_temperature = False    # Default arguments for "SpatialSoftmax"
        self.observation.encoder.rgb.core_kwargs.pool_kwargs.temperature = 1.0                # Default arguments for "SpatialSoftmax"
        self.observation.encoder.rgb.core_kwargs.pool_kwargs.noise_std = 0.0                  # Default arguments for "SpatialSoftmax"
        self.observation.encoder.rgb.core_kwargs.pool_kwargs.output_variance = False          # Default arguments for "SpatialSoftmax"
        self.observation.encoder.rgb.core_kwargs.pool_kwargs.do_not_lock_keys()

        # RGB: Obs Randomizer settings
        self.observation.encoder.rgb.obs_randomizer_class = None                  # Can set to 'CropRandomizer' to use crop randomization
        self.observation.encoder.rgb.obs_randomizer_kwargs.crop_height = 76       # Default arguments for "CropRandomizer"
        self.observation.encoder.rgb.obs_randomizer_kwargs.crop_width = 76        # Default arguments for "CropRandomizer"
        self.observation.encoder.rgb.obs_randomizer_kwargs.num_crops = 1          # Default arguments for "CropRandomizer"
        self.observation.encoder.rgb.obs_randomizer_kwargs.pos_enc = False        # Default arguments for "CropRandomizer"
        self.observation.encoder.rgb.obs_randomizer_kwargs.do_not_lock_keys()

        # Allow for other custom modalities to be specified
        self.observation.encoder.do_not_lock_keys()

        # =============== Depth default encoder (same as rgb) ===============
        self.observation.encoder.depth = deepcopy(self.observation.encoder.rgb)

        # =============== Scan default encoder (Conv1d backbone + linear layer output) ===============
        self.observation.encoder.scan = deepcopy(self.observation.encoder.rgb)
        self.observation.encoder.scan.core_kwargs.pop("backbone_class")
        self.observation.encoder.scan.core_kwargs.pop("backbone_kwargs")

        # Scan: Modify the core class + kwargs, otherwise, is same as rgb encoder
        self.observation.encoder.scan.core_class = "ScanCore"
        self.observation.encoder.scan.core_kwargs.conv_activation = "relu"
        self.observation.encoder.scan.core_kwargs.conv_kwargs.out_channels = [32, 64, 64]
        self.observation.encoder.scan.core_kwargs.conv_kwargs.kernel_size = [8, 4, 2]
        self.observation.encoder.scan.core_kwargs.conv_kwargs.stride = [4, 2, 1]

    @property
    def use_goals(self):
        # whether the agent is goal-conditioned
        return len([obs_key for modality in self.observation.modalities.goal.values() for obs_key in modality]) > 0

    @property
    def all_obs_keys(self):
        """
        This grabs the union of observation keys over all modalities (e.g.: low_dim, rgb, depth, etc.) and over all
        modality groups (e.g: obs, goal, subgoal, etc...)

        Returns:
            n-array: all observation keys used for this model
        """
        # pool all modalities
        return sorted(tuple(set([
            obs_key for group in [
                self.observation.modalities.obs.values(),
                self.observation.modalities.goal.values()
            ]
            for modality in group
            for obs_key in modality
         ])))
