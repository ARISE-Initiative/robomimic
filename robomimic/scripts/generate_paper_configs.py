"""
Helper script to generate jsons for reproducing paper experiments.

Args:
    config_dir (str): Directory where generated configs will be placed. 
        Defaults to 'paper' subfolder in exps folder of repository

    dataset_dir (str): Base dataset directory where released datasets can be
        found on disk. Defaults to datasets folder in repository.

    output_dir (str): Base output directory for all training runs that will be 
        written to generated configs.

Example usage:
    # Assume datasets alredy exist in robomimic/../datasets folder. Configs will be generated under robomimic/exps/paper
    python generate_paper_configs.py --output_dir /tmp/experiment_results

    # Specify where datasets exist, and specify where configs should be generated.
    python generate_paper_configs.py --config_dir /tmp/configs --dataset_dir /tmp/datasets --output_dir /tmp/experiment_results
"""
import os
import argparse
import robomimic
from robomimic import DATASET_REGISTRY
from robomimic.config import Config, BCConfig, BCQConfig, CQLConfig, HBCConfig, IRISConfig, config_factory


def modify_config_for_default_low_dim_exp(config):
    """
    Modifies a Config object with experiment, training, and observation settings that
    were used across all low-dimensional experiments by default.

    Args:
        config (Config instance): config to modify
    """

    with config.experiment.values_unlocked():
        # save model during every evaluation (every 50 epochs)
        config.experiment.save.enabled = True
        config.experiment.save.every_n_epochs = 50

        # every epoch is 100 gradient steps, and validation epoch is 10 gradient steps
        config.experiment.epoch_every_n_steps = 100
        config.experiment.validation_epoch_every_n_steps = 10

        # do 50 evaluation rollouts every 50 epochs
        # NOTE: horizon will generally get set depending on the task and dataset type
        config.experiment.rollout.enabled = True
        config.experiment.rollout.n = 50
        config.experiment.rollout.horizon = 400
        config.experiment.rollout.rate = 50
        config.experiment.rollout.warmstart = 0
        config.experiment.rollout.terminate_on_success = True

    with config.train.values_unlocked():
        # assume entire dataset can fit in memory
        config.train.num_data_workers = 0
        config.train.hdf5_cache_mode = "all"

        # batch size 100 and 2000 training epochs
        config.train.batch_size = 100
        config.train.num_epochs = 2000

    with config.observation.values_unlocked():
        # default observation is eef pose, gripper finger position, and object information,
        # all of which are low-dim. 
        default_low_dim_obs = [
            "robot0_eef_pos", 
            "robot0_eef_quat", 
            "robot0_gripper_qpos", 
            "object",
        ]
        # handle hierarchical observation configs
        if config.algo_name == "hbc":
            mod_configs_to_set = [
                config.observation.actor.modalities.obs,
                config.observation.planner.modalities.obs,
                config.observation.planner.modalities.subgoal,
            ]
        elif config.algo_name == "iris":
            mod_configs_to_set = [
                config.observation.actor.modalities.obs,
                config.observation.value_planner.planner.modalities.obs,
                config.observation.value_planner.planner.modalities.subgoal,
                config.observation.value_planner.value.modalities.obs,
            ]
        else:
            mod_configs_to_set = [config.observation.modalities.obs]
        # set all observations / subgoals to use the correct low-dim modalities
        for mod_config in mod_configs_to_set:
            mod_config.low_dim = list(default_low_dim_obs)
            mod_config.image = []

    return config


def modify_config_for_default_image_exp(config):
    """
    Modifies a Config object with experiment, training, and observation settings that
    were used across all image experiments by default.

    Args:
        config (Config instance): config to modify
    """
    assert config.algo_name not in ["hbc", "iris"], "no image training for HBC and IRIS"

    with config.experiment.values_unlocked():
        # save model during every evaluation (every 20 epochs)
        config.experiment.save.enabled = True
        config.experiment.save.every_n_epochs = 20

        # every epoch is 500 gradient steps, and validation epoch is 50 gradient steps
        config.experiment.epoch_every_n_steps = 500
        config.experiment.validation_epoch_every_n_steps = 50

        # do 50 evaluation rollouts every 20 epochs
        # NOTE: horizon will generally get set depending on the task and dataset type
        config.experiment.rollout.enabled = True
        config.experiment.rollout.n = 50
        config.experiment.rollout.horizon = 400
        config.experiment.rollout.rate = 20
        config.experiment.rollout.warmstart = 0
        config.experiment.rollout.terminate_on_success = True

    with config.train.values_unlocked():
        # only cache low-dim info, and use 2 data workers to increase fetch speed for image obs
        config.train.num_data_workers = 2
        config.train.hdf5_cache_mode = "low_dim"

        # batch size 16 and 600 training epochs
        config.train.batch_size = 16
        config.train.num_epochs = 600


    with config.observation.values_unlocked():
        # default low-dim observation is eef pose, gripper finger position
        # default image observation is external camera and wrist camera
        config.observation.modalities.obs.low_dim = [
            "robot0_eef_pos", 
            "robot0_eef_quat", 
            "robot0_gripper_qpos", 
        ]
        config.observation.modalities.obs.image = [
            "agentview_image",
            "robot0_eye_in_hand_image",
        ]
        config.observation.modalities.goal.low_dim = []
        config.observation.modalities.goal.image = []

        # default image encoder architecture is ResNet with spatial softmax
        config.observation.encoder.visual_core = 'ResNet18Conv'
        config.observation.encoder.visual_core_kwargs = Config()
        config.observation.encoder.visual_feature_dimension = 64

        config.observation.encoder.use_spatial_softmax = True
        config.observation.encoder.spatial_softmax_kwargs.num_kp = 32
        config.observation.encoder.spatial_softmax_kwargs.learnable_temperature = False
        config.observation.encoder.spatial_softmax_kwargs.temperature = 1.0
        config.observation.encoder.spatial_softmax_kwargs.noise_std = 0.

        # use crop randomization as well
        config.observation.encoder.obs_randomizer_class = 'CropRandomizer'  # observation randomizer class
        config.observation.encoder.obs_randomizer_kwargs.crop_height = 76
        config.observation.encoder.obs_randomizer_kwargs.crop_width = 76
        config.observation.encoder.obs_randomizer_kwargs.num_crops = 1
        config.observation.encoder.obs_randomizer_kwargs.pos_enc = False

    return config


def modify_config_for_dataset(config, task_name, dataset_type, hdf5_type, base_dataset_dir, filter_key=None):
    """
    Modifies a Config object with experiment, training, and observation settings to
    correspond to experiment settings for the dataset collected on @task_name with
    dataset source @dataset_type (e.g. ph, mh, mg), and hdf5 type @hdf5_type (e.g. low_dim
    or image).

    Args:
        config (Config instance): config to modify

        task_name (str): identify task that dataset was collected on

        dataset_type (str): dataset type for this dataset (e.g. ph, mh, mg).

        hdf5_type (str): hdf5 type for this dataset (e.g. raw, low_dim, image). 

        base_dataset_dir (str): path to directory where datasets are on disk.
            Directory structure is expected to be consistent with the output
            of @make_dataset_dirs in the download_datasets.py script.

        filter_key (str): if not None, use the provided filter key to select a subset of the
            provided dataset
    """
    assert task_name in DATASET_REGISTRY, \
        "task {} not found in dataset registry!".format(task_name)
    assert dataset_type in DATASET_REGISTRY[task_name], \
        "dataset type {} not found for task {} in dataset registry!".format(dataset_type, task_name)
    assert hdf5_type in DATASET_REGISTRY[task_name][dataset_type], \
        "hdf5 type {} not found for dataset type {} and task {} in dataset registry!".format(hdf5_type, dataset_type, task_name)

    is_real_dataset = "real" in task_name
    if is_real_dataset:
        assert config.algo_name == "bc", "we only ran BC-RNN on real robot"
    else:
        assert hdf5_type != "raw", "cannot train on raw demonstrations"

    with config.experiment.values_unlocked():

        # look up rollout evaluation horizon in registry and set it
        config.experiment.rollout.horizon = DATASET_REGISTRY[task_name][dataset_type][hdf5_type]["horizon"]

        if dataset_type == "mg":
            # machine-generated datasets did not use validation
            config.experiment.validate = False

        if is_real_dataset:
            # no evaluation rollouts for real robot training
            config.experiment.rollout.enabled = False

    with config.train.values_unlocked():
        # set dataset path and possibly filter key
        file_name = DATASET_REGISTRY[task_name][dataset_type][hdf5_type]["url"].split("/")[-1]
        config.train.data = os.path.join(base_dataset_dir, task_name, dataset_type, file_name)
        if filter_key is not None:
            config.train.hdf5_filter_key = filter_key

    with config.observation.values_unlocked():
        # maybe modify observation names and randomization sizes (since image size might be different)

        if is_real_dataset:
            # modify observation names for real robot datasets
            config.observation.modalities.obs.low_dim = [
                "ee_pose", 
                "gripper_position", 
            ]

            if task_name == "tool_hang_real":
                # side and wrist camera
                config.observation.modalities.obs.image = [
                    "image_side",
                    "image_wrist",
                ]
                # 240x240 images -> crops should be 216x216
                config.observation.encoder.obs_randomizer_kwargs.crop_height = 216
                config.observation.encoder.obs_randomizer_kwargs.crop_width = 216
            else:
                # front and wrist camera
                config.observation.modalities.obs.image = [
                    "image",
                    "image_wrist",
                ]
                # 120x120 images -> crops should be 108x108
                config.observation.encoder.obs_randomizer_kwargs.crop_height = 108
                config.observation.encoder.obs_randomizer_kwargs.crop_width = 108

        elif hdf5_type in ["image", "image_sparse", "image_dense"]:
            if task_name == "transport":
                # robot proprioception per arm
                config.observation.modalities.obs.low_dim = [
                    "robot0_eef_pos", 
                    "robot0_eef_quat", 
                    "robot0_gripper_qpos", 
                    "robot1_eef_pos", 
                    "robot1_eef_quat", 
                    "robot1_gripper_qpos", 
                ]

                # shoulder and wrist cameras per arm
                config.observation.modalities.obs.image = [
                    "shouldercamera0_image",
                    "robot0_eye_in_hand_image",
                    "shouldercamera1_image",
                    "robot1_eye_in_hand_image",
                ]
            elif task_name == "tool_hang":
                # side and wrist camera
                config.observation.modalities.obs.image = [
                    "sideview_image",
                    "robot0_eye_in_hand_image",
                ]
                # 240x240 images -> crops should be 216x216
                config.observation.encoder.obs_randomizer_kwargs.crop_height = 216
                config.observation.encoder.obs_randomizer_kwargs.crop_width = 216

        elif hdf5_type in ["low_dim", "low_dim_sparse", "low_dim_dense"]:
            if task_name == "transport":
                # robot proprioception per arm
                default_low_dim_obs = [
                    "robot0_eef_pos", 
                    "robot0_eef_quat", 
                    "robot0_gripper_qpos", 
                    "robot1_eef_pos", 
                    "robot1_eef_quat", 
                    "robot1_gripper_qpos", 
                    "object",
                ]
                # handle hierarchical observation configs
                if config.algo_name == "hbc":
                    mod_configs_to_set = [
                        config.observation.actor.modalities.obs,
                        config.observation.planner.modalities.obs,
                        config.observation.planner.modalities.subgoal,
                    ]
                elif config.algo_name == "iris":
                    mod_configs_to_set = [
                        config.observation.actor.modalities.obs,
                        config.observation.value_planner.planner.modalities.obs,
                        config.observation.value_planner.planner.modalities.subgoal,
                        config.observation.value_planner.value.modalities.obs,
                    ]
                else:
                    mod_configs_to_set = [config.observation.modalities.obs]
                # set all observations / subgoals to use the correct low-dim modalities
                for mod_config in mod_configs_to_set:
                    mod_config.low_dim = list(default_low_dim_obs)
                    mod_config.image = []

    return config


def modify_bc_config_for_dataset(config, task_name, dataset_type, hdf5_type):
    """
    Modifies a BCConfig object for training on a particular kind of dataset. This function
    just sets algorithm hyperparameters in the algo config depending on the kind of 
    dataset.

    Args:
        config (BCConfig instance): config to modify

        task_name (str): identify task that dataset was collected on. Only used to distinguish
            between simulation and real-world, for an assert statement

        dataset_type (str): dataset type for this dataset (e.g. ph, mh, mg, paired).

        hdf5_type (str): hdf5 type for this dataset (e.g. raw, low_dim, image). 
    """
    assert isinstance(config, BCConfig), "must be BCConfig"
    assert config.algo_name == "bc", "must be BCConfig"
    assert dataset_type in ["ph", "mh", "mg", "paired"], "invalid dataset type"
    is_real_dataset = "real" in task_name
    if not is_real_dataset:
        assert hdf5_type != "raw", "cannot train on raw demonstrations"

    with config.algo.values_unlocked():
        # base parameters that may get modified
        config.algo.optim_params.policy.learning_rate.initial = 1e-4            # learning rate 1e-4
        config.algo.actor_layer_dims = (1024, 1024)                             # MLP size (1024, 1024)
        config.algo.gmm.enabled = True                                          # enable GMM

        if dataset_type == "mg":
            # machine-generated datasets don't use GMM
            config.algo.gmm.enabled = False                                     # disable GMM
            if hdf5_type in ["low_dim", "low_dim_sparse", "low_dim_dense"]:
                # low-dim mg uses LR 1e-3
                config.algo.optim_params.policy.learning_rate.initial = 1e-3    # learning rate 1e-3

    return config


def modify_bc_rnn_config_for_dataset(config, task_name, dataset_type, hdf5_type):
    """
    Modifies a BCConfig object for training on a particular kind of dataset. This function
    just sets algorithm hyperparameters in the algo config depending on the kind of 
    dataset.

    Args:
        config (BCConfig instance): config to modify

        task_name (str): identify task that dataset was collected on. Only used to distinguish
            between simulation and real-world, for an assert statement

        dataset_type (str): dataset type for this dataset (e.g. ph, mh, mg, paired).

        hdf5_type (str): hdf5 type for this dataset (e.g. raw, low_dim, image). 
    """
    assert isinstance(config, BCConfig), "must be BCConfig"
    assert config.algo_name == "bc", "must be BCConfig"
    assert dataset_type in ["ph", "mh", "mg", "paired"], "invalid dataset type"
    is_real_dataset = "real" in task_name
    if not is_real_dataset:
        assert hdf5_type != "raw", "cannot train on raw demonstrations"

    with config.train.values_unlocked():
        # make sure RNN is enabled with sequence length 10
        config.train.seq_length = 10

    with config.algo.values_unlocked():
        # make sure RNN is enabled with sequence length 10
        config.algo.rnn.enabled = True
        config.algo.rnn.horizon = 10

        # base parameters that may get modified
        config.algo.optim_params.policy.learning_rate.initial = 1e-4            # learning rate 1e-4
        config.algo.actor_layer_dims = ()                                       # no MLP layers between rnn layer and output
        config.algo.gmm.enabled = True                                          # enable GMM
        config.algo.rnn.hidden_dim = 400                                        # rnn dim 400

        if dataset_type == "mg":
            # update hyperparams for machine-generated datasets
            config.algo.gmm.enabled = False                                     # disable GMM
            if hdf5_type not in ["low_dim", "low_dim_sparse", "low_dim_dense"]:
                # image datasets use RNN dim 1000
                config.algo.rnn.hidden_dim = 1000                               # rnn dim 1000
        else:
            # update hyperparams for all other dataset types (ph, mh, paired)
            if hdf5_type not in ["low_dim", "low_dim_sparse", "low_dim_dense"]:
                # image datasets use RNN dim 1000
                config.algo.rnn.hidden_dim = 1000                               # rnn dim 1000

    return config


def modify_bcq_config_for_dataset(config, task_name, dataset_type, hdf5_type):
    """
    Modifies a BCQConfig object for training on a particular kind of dataset. This function
    just sets algorithm hyperparameters in the algo config depending on the kind of 
    dataset.

    Args:
        config (BCQConfig instance): config to modify

        task_name (str): identify task that dataset was collected on. Only used to distinguish
            between simulation and real-world, for an assert statement

        dataset_type (str): dataset type for this dataset (e.g. ph, mh, mg, paired).

        hdf5_type (str): hdf5 type for this dataset (e.g. raw, low_dim, image). 
    """
    assert isinstance(config, BCQConfig), "must be BCQConfig"
    assert config.algo_name == "bcq", "must be BCQConfig"
    assert dataset_type in ["ph", "mh", "mg", "paired"], "invalid dataset type"
    is_real_dataset = "real" in task_name
    assert not is_real_dataset, "we only ran BC-RNN on real robot"
    if not is_real_dataset:
        assert hdf5_type != "raw", "cannot train on raw demonstrations"

    with config.algo.values_unlocked():
        # base parameters that may get modified further
        config.algo.optim_params.critic.learning_rate.initial = 1e-4                # all learning rates 1e-3
        config.algo.optim_params.action_sampler.learning_rate.initial = 1e-4
        config.algo.optim_params.actor.learning_rate.initial = 1e-3
        config.algo.actor.enabled = False                                           # disable actor by default
        config.algo.action_sampler.vae.enabled = True                               # use VAE action sampler
        config.algo.action_sampler.gmm.enabled = False
        config.algo.action_sampler.vae.kl_weight = 0.05                             # beta 0.05 for VAE
        config.algo.action_sampler.vae.latent_dim = 14                              # latent dim 14
        config.algo.action_sampler.vae.prior.learn = False                          # N(0, 1) prior
        config.algo.critic.layer_dims = (300, 400)                                  # all MLP sizes at (300, 400)
        config.algo.action_sampler.vae.encoder_layer_dims = (300, 400)
        config.algo.action_sampler.vae.decoder_layer_dims = (300, 400)
        config.algo.actor.layer_dims = (300, 400)
        config.algo.target_tau = 5e-4                                               # tau 5e-4
        config.algo.discount = 0.99                                                 # discount 0.99
        config.algo.critic.num_action_samples = 10                                  # number of action sampler samples at train and test
        config.algo.critic.num_action_samples_rollout = 100

        if dataset_type == "mg":
            # update hyperparams for machine-generated datasets
            config.algo.optim_params.critic.learning_rate.initial = 1e-3            # all learning rates 1e-3
            config.algo.optim_params.action_sampler.learning_rate.initial = 1e-3
            config.algo.optim_params.actor.learning_rate.initial = 1e-3
            config.algo.action_sampler.vae.kl_weight = 0.5                          # beta 0.5 for VAE
            config.algo.target_tau = 5e-3                                           # tau 5e-3

            if hdf5_type in ["low_dim", "low_dim_sparse", "low_dim_dense"]:
                # enable actor only on low-dim
                config.algo.actor.enabled = True
        else:
            # make some modifications where needed for human datasets
            if hdf5_type in ["low_dim", "low_dim_sparse", "low_dim_dense"]:
                if dataset_type in ["mh", "paired"]:
                    # low-dim, MH had higher layer sizes
                    config.algo.critic.layer_dims = (1024, 1024)
                    config.algo.action_sampler.vae.encoder_layer_dims = (1024, 1024)
                    config.algo.action_sampler.vae.decoder_layer_dims = (1024, 1024)
                    config.algo.action_sampler.vae.prior_layer_dims = (1024, 1024)

                    config.algo.action_sampler.vae.kl_weight = 0.5

                    # use learned GMM prior for MH dataset
                    config.algo.action_sampler.vae.prior.learn = True
                    config.algo.action_sampler.vae.prior.is_conditioned = True
                    config.algo.action_sampler.vae.prior.use_gmm = True
                    config.algo.action_sampler.vae.prior.gmm_learn_weights = True
            else:
                if dataset_type == "ph":
                    # image, PH used higher critic LR of 1e-3
                    config.algo.optim_params.critic.learning_rate.initial = 1e-3
                # image datasets used bigger VAE
                config.algo.action_sampler.vae.encoder_layer_dims = (1024, 1024)
                config.algo.action_sampler.vae.decoder_layer_dims = (1024, 1024)
                if dataset_type in ["mh", "paired"]:
                    # image, MH also had bigger critic
                    config.algo.critic.layer_dims = (1024, 1024)

    return config


def modify_cql_config_for_dataset(config, task_name, dataset_type, hdf5_type):
    """
    Modifies a CQLConfig object for training on a particular kind of dataset. This function
    just sets algorithm hyperparameters in the algo config depending on the kind of 
    dataset.

    Args:
        config (CQLConfig instance): config to modify

        task_name (str): identify task that dataset was collected on. Only used to distinguish
            between simulation and real-world, for an assert statement

        dataset_type (str): dataset type for this dataset (e.g. ph, mh, mg, paired).

        hdf5_type (str): hdf5 type for this dataset (e.g. raw, low_dim, image). 
    """
    assert isinstance(config, CQLConfig), "must be CQLConfig"
    assert config.algo_name == "cql", "must be CQLConfig"
    assert dataset_type in ["ph", "mh", "mg", "paired"], "invalid dataset type"
    is_real_dataset = "real" in task_name
    assert not is_real_dataset, "we only ran BC-RNN on real robot"
    if not is_real_dataset:
        assert hdf5_type != "raw", "cannot train on raw demonstrations"

    with config.train.values_unlocked():
        # CQL uses batch size 1024 (for low-dim) and 8 (for image)
        if hdf5_type in ["low_dim", "low_dim_sparse", "low_dim_dense"]:
            config.train.batch_size = 1024
        else:
            config.train.batch_size = 8

    with config.algo.values_unlocked():
        # base parameters that may get modified further
        config.algo.optim_params.critic.learning_rate.initial = 1e-3                # learning rates
        config.algo.optim_params.actor.learning_rate.initial = 3e-4
        config.algo.actor.target_entropy = "default"                                # use automatic entropy tuning to default target value
        config.algo.critic.deterministic_backup = True                              # deterministic Q-backup
        config.algo.critic.target_q_gap = 5.0                                       # use Lagrange, with threshold 5.0
        config.algo.critic.min_q_weight = 1.0
        config.algo.target_tau = 5e-3                                               # tau 5e-3
        config.algo.discount = 0.99                                                 # discount 0.99
        config.algo.critic.layer_dims = (300, 400)                                  # all MLP sizes at (300, 400)
        config.algo.actor.layer_dims = (300, 400)

        if hdf5_type not in ["low_dim", "low_dim_sparse", "low_dim_dense"]:
            # update policy LR to 1e-4 for image runs
            config.algo.optim_params.actor.learning_rate.initial = 1e-4

    return config


def modify_hbc_config_for_dataset(config, task_name, dataset_type, hdf5_type):
    """
    Modifies a HBCConfig object for training on a particular kind of dataset. This function
    just sets algorithm hyperparameters in the algo config depending on the kind of 
    dataset.

    Args:
        config (HBCConfig instance): config to modify

        task_name (str): identify task that dataset was collected on. Only used to distinguish
            between simulation and real-world, for an assert statement

        dataset_type (str): dataset type for this dataset (e.g. ph, mh, mg, paired).

        hdf5_type (str): hdf5 type for this dataset (e.g. raw, low_dim, image). 
    """
    assert isinstance(config, HBCConfig), "must be HBCConfig"
    assert config.algo_name == "hbc", "must be HBCConfig"
    assert dataset_type in ["ph", "mh", "mg", "paired"], "invalid dataset type"
    assert hdf5_type in ["low_dim", "low_dim_sparse", "low_dim_dense"], "HBC only runs on low-dim"
    is_real_dataset = "real" in task_name
    assert not is_real_dataset, "we only ran BC-RNN on real robot"

    with config.algo.values_unlocked():
        # base parameters that may get modified further
        config.algo.actor.optim_params.policy.learning_rate.initial = 1e-3          # learning rates
        config.algo.planner.optim_params.goal_network.learning_rate.initial = 1e-3

        config.algo.planner.vae.enabled = True                                      # goal VAE settings
        config.algo.planner.vae.kl_weight = 5e-4                                    # beta 5e-4
        config.algo.planner.vae.latent_dim = 16                                     # latent dim 16
        config.algo.planner.vae.prior.learn = True                                  # learn GMM prior with 10 modes
        config.algo.planner.vae.prior.is_conditioned = True
        config.algo.planner.vae.prior.use_gmm = True
        config.algo.planner.vae.prior.gmm_learn_weights = True
        config.algo.planner.vae.prior.gmm_num_modes = 10
        config.algo.planner.vae.encoder_layer_dims = (1024, 1024)                   # VAE network sizes
        config.algo.planner.vae.decoder_layer_dims = (1024, 1024)
        config.algo.planner.vae.prior_layer_dims = (1024, 1024)

        config.algo.actor.rnn.hidden_dim = 400                                      # actor RNN dim
        config.algo.actor.actor_layer_dims = ()                                     # no MLP layers between rnn layer and output

        if dataset_type == "mg":
            # update hyperparams for machine-generated datasets
            config.algo.actor.rnn.hidden_dim = 100
            config.algo.actor.actor_layer_dims = (1024, 1024)

    return config


def modify_iris_config_for_dataset(config, task_name, dataset_type, hdf5_type):
    """
    Modifies a IRISConfig object for training on a particular kind of dataset. This function
    just sets algorithm hyperparameters in the algo config depending on the kind of 
    dataset.

    Args:
        config (IRISConfig instance): config to modify

        task_name (str): identify task that dataset was collected on. Only used to distinguish
            between simulation and real-world, for an assert statement

        dataset_type (str): dataset type for this dataset (e.g. ph, mh, mg, paired).

        hdf5_type (str): hdf5 type for this dataset (e.g. raw, low_dim, image). 
    """
    assert isinstance(config, IRISConfig), "must be IRISConfig"
    assert config.algo_name == "iris", "must be IRISConfig"
    assert dataset_type in ["ph", "mh", "mg", "paired"], "invalid dataset type"
    assert hdf5_type in ["low_dim", "low_dim_sparse", "low_dim_dense"], "IRIS only runs on low-dim"
    is_real_dataset = "real" in task_name
    assert not is_real_dataset, "we only ran BC-RNN on real robot"

    with config.algo.values_unlocked():
        # base parameters that may get modified further
        config.algo.actor.optim_params.policy.learning_rate.initial = 1e-3                          # learning rates
        config.algo.value_planner.planner.optim_params.goal_network.learning_rate.initial = 1e-3
        config.algo.value_planner.value.optim_params.critic.learning_rate.initial = 1e-3
        config.algo.value_planner.value.optim_params.action_sampler.learning_rate.initial = 1e-4

        config.algo.value_planner.planner.vae.enabled = True                                        # goal VAE settings
        config.algo.value_planner.planner.vae.kl_weight = 5e-4                                      # beta 5e-4
        config.algo.value_planner.planner.vae.latent_dim = 14                                       # latent dim 14
        config.algo.value_planner.planner.vae.prior.learn = True                                    # learn GMM prior with 10 modes
        config.algo.value_planner.planner.vae.prior.is_conditioned = True
        config.algo.value_planner.planner.vae.prior.use_gmm = True
        config.algo.value_planner.planner.vae.prior.gmm_learn_weights = True
        config.algo.value_planner.planner.vae.prior.gmm_num_modes = 10
        config.algo.value_planner.planner.vae.encoder_layer_dims = (1024, 1024)                     # VAE network sizes
        config.algo.value_planner.planner.vae.decoder_layer_dims = (1024, 1024)
        config.algo.value_planner.planner.vae.prior_layer_dims = (1024, 1024)

        config.algo.value_planner.value.target_tau = 5e-4                                           # Value tau
        config.algo.value_planner.value.action_sampler.vae.kl_weight = 0.5                          # Value KL
        config.algo.value_planner.value.action_sampler.vae.latent_dim = 16
        config.algo.value_planner.value.action_sampler.actor_layer_dims = (300, 400)

        config.algo.actor.rnn.hidden_dim = 400                                                      # actor RNN dim
        config.algo.actor.actor_layer_dims = ()                                                     # no MLP layers between rnn layer and output

        if dataset_type in ["mh", "paired"]:
            # value LR 1e-4, KL weight is 0.05 for multi-human datasets
            config.algo.value_planner.value.optim_params.critic.learning_rate.initial = 1e-4
            config.algo.value_planner.value.action_sampler.vae.kl_weight = 0.05

        if dataset_type in ["mg"]:
            # Enable value actor and set larger target tau
            config.algo.value_planner.value.actor.enabled = True
            config.algo.value_planner.value.optim_params.actor.learning_rate.initial = 1e-3
            config.algo.value_planner.value.target_tau = 5e-3

    return config


def generate_experiment_config(
    base_exp_name, 
    base_config_dir, 
    base_dataset_dir, 
    base_output_dir, 
    algo_name, 
    algo_config_modifier, 
    task_name, 
    dataset_type, 
    hdf5_type,
    filter_key=None,
    additional_name=None,
    additional_config_modifier=None,
):
    """
    Helper function to generate a config for a particular experiment.

    Args:
        base_exp_name (str): name that identifies this set of experiments

        base_config_dir (str): base directory to place generated configs

        base_dataset_dir (str): path to directory where datasets are on disk.
            Directory structure is expected to be consistent with the output
            of @make_dataset_dirs in the download_datasets.py script.

        base_output_dir (str): directory to save training results to. If None, will use the directory
            from the default algorithm configs.

        algo_name (str): identifies the algorithm - one of ["bc", "bc_rnn", "bcq", "cql", hbc", "iris"]

        algo_config_modifier (function): function to modify config to add algo hyperparameter
            settings, given the task, dataset, and hdf5 types.

        task_name (str): identify task that dataset was collected on. Only used to distinguish
            between simulation and real-world, for an assert statement

        dataset_type (str): dataset type for this dataset (e.g. ph, mh, mg, paired).

        hdf5_type (str): hdf5 type for this dataset (e.g. raw, low_dim, image). 

        filter_key (str): if not None, use the provided filter key to select a subset of the
            provided dataset

        additional_name (str): if provided, will add this name to the generated experiment name, and
            the name of the generated config json

        additional_config_modifier (function): if provided, run this last function on the config
            to make final modifications before generating the json.
    """
    if "real" not in task_name:
        assert hdf5_type != "raw", "cannot train on raw demonstrations"

    # decide whether to use low-dim or image training defaults
    modifier_for_obs = modify_config_for_default_image_exp
    if hdf5_type in ["low_dim", "low_dim_sparse", "low_dim_dense"]:
        modifier_for_obs = modify_config_for_default_low_dim_exp

    algo_config_name = "bc" if algo_name == "bc_rnn" else algo_name
    config = config_factory(algo_name=algo_config_name)
    # turn into default config for observation type (low-dim or image)
    config = modifier_for_obs(config)
    # add in config based on the dataset
    config = modify_config_for_dataset(
        config=config, 
        task_name=task_name, 
        dataset_type=dataset_type, 
        hdf5_type=hdf5_type, 
        base_dataset_dir=base_dataset_dir,
        filter_key=filter_key,
    )
    # add in algo hypers based on dataset
    config = algo_config_modifier(
        config=config, 
        task_name=task_name, 
        dataset_type=dataset_type, 
        hdf5_type=hdf5_type,
    )
    if additional_config_modifier is not None:
        # use additional config modifier if provided
        config = additional_config_modifier(config)

    # account for filter key in experiment naming and directory naming
    filter_key_str = "_{}".format(filter_key) if filter_key is not None else ""
    dataset_type_dir = "{}/{}".format(dataset_type, filter_key) if filter_key is not None else dataset_type

    # account for @additional_name
    additional_name_str = "_{}".format(additional_name) if additional_name is not None else ""
    json_name = "{}{}".format(algo_name, additional_name_str)

    # set experiment name
    with config.experiment.values_unlocked():
        config.experiment.name = "{}_{}_{}_{}{}_{}{}".format(base_exp_name, algo_name, task_name, dataset_type, filter_key_str, hdf5_type, additional_name_str)
    # set output folder
    with config.train.values_unlocked():
        if base_output_dir is None:
            base_output_dir = config.train.output_dir
        config.train.output_dir = os.path.join(base_output_dir, base_exp_name, algo_name, task_name, dataset_type_dir, hdf5_type, "trained_models")
    
    # save config to json file
    dir_to_save = os.path.join(base_config_dir, base_exp_name, task_name, dataset_type_dir, hdf5_type)
    os.makedirs(dir_to_save, exist_ok=True)
    json_path = os.path.join(dir_to_save, "{}.json".format(json_name))
    config.dump(filename=json_path)

    return config, json_path


def generate_core_configs(
    base_config_dir, 
    base_dataset_dir, 
    base_output_dir, 
    algo_to_config_modifier, 
):
    """
    Helper function to generate all configs for core set of experiments.

    Args:
        base_config_dir (str): base directory to place generated configs

        base_dataset_dir (str): path to directory where datasets are on disk.
            Directory structure is expected to be consistent with the output
            of @make_dataset_dirs in the download_datasets.py script.

        base_output_dir (str): directory to save training results to. If None, will use the directory
            from the default algorithm configs.

        algo_to_config_modifier (dict): dictionary that maps algo name to a function that modifies configs 
            to add algo hyperparameter settings, given the task, dataset, and hdf5 types.
    """
    core_json_paths = Config() # use for convenient nested dict
    for task in DATASET_REGISTRY:
        for dataset_type in DATASET_REGISTRY[task]:
            for hdf5_type in DATASET_REGISTRY[task][dataset_type]:
                # if not real robot dataset, skip raw hdf5
                is_real_dataset = ("real" in task)
                if not is_real_dataset and hdf5_type == "raw":
                    continue
                
                # get list of algorithms to generate configs for, for this hdf5 dataset
                algos_to_generate = ["bc", "bc_rnn", "bcq", "cql", "hbc", "iris"]
                if hdf5_type not in ["low_dim", "low_dim_sparse", "low_dim_dense"]:
                    # no hbc or iris for image runs
                    algos_to_generate = algos_to_generate[:-2]
                if is_real_dataset:
                    # we only ran BC-RNN on real robot
                    algos_to_generate = ["bc_rnn"]

                for algo_name in algos_to_generate:

                    # generate config for this experiment
                    config, json_path = generate_experiment_config(
                        base_exp_name="core",
                        base_config_dir=base_config_dir,
                        base_dataset_dir=base_dataset_dir,
                        base_output_dir=base_output_dir,
                        algo_name=algo_name, 
                        algo_config_modifier=algo_to_config_modifier[algo_name], 
                        task_name=task, 
                        dataset_type=dataset_type, 
                        hdf5_type=hdf5_type,
                    )

                    # save json path into dict
                    core_json_paths[task][dataset_type][hdf5_type][algo_name] = json_path

    return core_json_paths


def generate_subopt_configs(
    base_config_dir, 
    base_dataset_dir, 
    base_output_dir, 
    algo_to_config_modifier, 
):
    """
    Helper function to generate all configs for the suboptimal human subsets of the multi-human datasets.
    Note that while the paper includes the results on the can-paired dataset along with results on these
    datasets, the configs for runs on the can-paired dataset is in the "core" set of runs.

    Args:
        base_config_dir (str): base directory to place generated configs

        base_dataset_dir (str): path to directory where datasets are on disk.
            Directory structure is expected to be consistent with the output
            of @make_dataset_dirs in the download_datasets.py script.

        base_output_dir (str): directory to save training results to. If None, will use the directory
            from the default algorithm configs.

        algo_to_config_modifier (dict): dictionary that maps algo name to a function that modifies configs 
            to add algo hyperparameter settings, given the task, dataset, and hdf5 types.
    """
    subopt_json_paths = Config() # use for convenient nested dict
    for task in ["lift", "can", "square", "transport"]:
        # only generate configs for multi-human data subsets
        for dataset_type in ["mh"]:
            # only low-dim / image
            for hdf5_type in ["low_dim", "image"]:

                # get list of algorithms to generate configs for, for this hdf5 dataset
                algos_to_generate = ["bc", "bc_rnn", "bcq", "cql", "hbc", "iris"]
                if hdf5_type == "image":
                    # no hbc or iris for image runs
                    algos_to_generate = algos_to_generate[:-2]

                for algo_name in algos_to_generate:

                    for fk in ["worse", "okay", "better", "worse_okay", "worse_better", "okay_better"]:

                        # generate config for this experiment
                        config, json_path = generate_experiment_config(
                            base_exp_name="subopt",
                            base_config_dir=base_config_dir,
                            base_dataset_dir=base_dataset_dir,
                            base_output_dir=base_output_dir,
                            algo_name=algo_name, 
                            algo_config_modifier=algo_to_config_modifier[algo_name], 
                            task_name=task, 
                            dataset_type=dataset_type, 
                            hdf5_type=hdf5_type,
                            filter_key=fk,
                        )

                        # save json path into dict
                        dataset_type_dir = "{}/{}".format(dataset_type, fk)
                        subopt_json_paths[task][dataset_type_dir][hdf5_type][algo_name] = json_path

    return subopt_json_paths


def generate_dataset_size_configs(
    base_config_dir, 
    base_dataset_dir, 
    base_output_dir, 
    algo_to_config_modifier, 
):
    """
    Helper function to generate all configs for the dataset size ablation experiments, where BC-RNN models
    were trained on 20% and 50% dataset sizes.

    Args:
        base_config_dir (str): base directory to place generated configs

        base_dataset_dir (str): path to directory where datasets are on disk.
            Directory structure is expected to be consistent with the output
            of @make_dataset_dirs in the download_datasets.py script.

        base_output_dir (str): directory to save training results to. If None, will use the directory
            from the default algorithm configs.

        algo_to_config_modifier (dict): dictionary that maps algo name to a function that modifies configs 
            to add algo hyperparameter settings, given the task, dataset, and hdf5 types.
    """
    size_ablation_json_paths = Config() # use for convenient nested dict
    for task in ["lift", "can", "square", "transport"]:
        for dataset_type in ["ph", "mh"]:
            for hdf5_type in ["low_dim", "image"]:

                # only bc-rnn
                algo_name = "bc_rnn"
                for fk in ["20_percent", "50_percent"]:

                    # generate config for this experiment
                    config, json_path = generate_experiment_config(
                        base_exp_name="dataset_size",
                        base_config_dir=base_config_dir,
                        base_dataset_dir=base_dataset_dir,
                        base_output_dir=base_output_dir,
                        algo_name=algo_name, 
                        algo_config_modifier=algo_to_config_modifier[algo_name], 
                        task_name=task, 
                        dataset_type=dataset_type, 
                        hdf5_type=hdf5_type,
                        filter_key=fk,
                    )

                    # save json path into dict
                    dataset_type_dir = "{}/{}".format(dataset_type, fk)
                    size_ablation_json_paths[task][dataset_type_dir][hdf5_type][algo_name] = json_path

    return size_ablation_json_paths


def generate_obs_ablation_configs(
    base_config_dir, 
    base_dataset_dir, 
    base_output_dir, 
    algo_to_config_modifier, 
):
    """
    Helper function to generate all configs for the observation ablation experiments, where BC and BC-RNN models
    were trained on different versions of low-dim and image observations.

    Args:
        base_config_dir (str): base directory to place generated configs

        base_dataset_dir (str): path to directory where datasets are on disk.
            Directory structure is expected to be consistent with the output
            of @make_dataset_dirs in the download_datasets.py script.

        base_output_dir (str): directory to save training results to. If None, will use the directory
            from the default algorithm configs.

        algo_to_config_modifier (dict): dictionary that maps algo name to a function that modifies configs 
            to add algo hyperparameter settings, given the task, dataset, and hdf5 types.
    """

    # observation config modifiers for these experiments
    def add_eef_vel(config):
        with config.observation.values_unlocked():
            old_low_dim_mods = list(config.observation.modalities.obs.low_dim)
            old_low_dim_mods.extend(["robot0_eef_vel_lin", "robot0_eef_vel_ang", "robot0_gripper_qvel"])
            if "robot1_eef_pos" in old_low_dim_mods:
                old_low_dim_mods.extend(["robot1_eef_vel_lin", "robot1_eef_vel_ang", "robot1_gripper_qvel"])
            config.observation.modalities.obs.low_dim = old_low_dim_mods
        return config

    def add_proprio(config):
        with config.observation.values_unlocked():
            old_low_dim_mods = list(config.observation.modalities.obs.low_dim)
            old_low_dim_mods.extend(["robot0_joint_pos_cos", "robot0_joint_pos_sin", "robot0_joint_vel"])
            if "robot1_eef_pos" in old_low_dim_mods:
                old_low_dim_mods.extend(["robot1_joint_pos_cos", "robot1_joint_pos_sin", "robot1_joint_vel"])
            config.observation.modalities.obs.low_dim = old_low_dim_mods
        return config

    def remove_wrist(config):
        with config.observation.values_unlocked():
            old_image_mods = list(config.observation.modalities.obs.image)
            config.observation.modalities.obs.image = [m for m in old_image_mods if "eye_in_hand" not in m]
        return config

    def remove_rand(config):
        with config.observation.values_unlocked():
            config.observation.encoder.obs_randomizer_class = None
        return config

    obs_ablation_json_paths = Config() # use for convenient nested dict
    for task in ["square", "transport"]:
        for dataset_type in ["ph", "mh"]:
            for hdf5_type in ["low_dim", "image"]:

                # observation modifiers to apply
                if hdf5_type == "low_dim":
                    obs_modifiers = [add_eef_vel, add_proprio]
                else:
                    obs_modifiers = [add_eef_vel, add_proprio, remove_wrist, remove_rand]

                # only bc and bc-rnn
                algos_to_generate = ["bc", "bc_rnn"]
                for algo_name in algos_to_generate:
                    for obs_modifier in obs_modifiers:
                        # generate config for this experiment
                        config, json_path = generate_experiment_config(
                            base_exp_name="obs_ablation",
                            base_config_dir=base_config_dir,
                            base_dataset_dir=base_dataset_dir,
                            base_output_dir=base_output_dir,
                            algo_name=algo_name, 
                            algo_config_modifier=algo_to_config_modifier[algo_name], 
                            task_name=task, 
                            dataset_type=dataset_type, 
                            hdf5_type=hdf5_type,
                            additional_name=obs_modifier.__name__,
                            additional_config_modifier=obs_modifier,
                        )

                        # save json path into dict
                        algo_name_str = "{}_{}".format(algo_name, obs_modifier.__name__)
                        obs_ablation_json_paths[task][dataset_type][hdf5_type][algo_name_str] = json_path

    return obs_ablation_json_paths


def generate_hyper_ablation_configs(
    base_config_dir, 
    base_dataset_dir, 
    base_output_dir, 
    algo_to_config_modifier, 
):
    """
    Helper function to generate all configs for the hyperparameter sensitivity experiments, 
    where BC-RNN models were trained on different ablations.

    Args:
        base_config_dir (str): base directory to place generated configs

        base_dataset_dir (str): path to directory where datasets are on disk.
            Directory structure is expected to be consistent with the output
            of @make_dataset_dirs in the download_datasets.py script.

        base_output_dir (str): directory to save training results to. If None, will use the directory
            from the default algorithm configs.

        algo_to_config_modifier (dict): dictionary that maps algo name to a function that modifies configs 
            to add algo hyperparameter settings, given the task, dataset, and hdf5 types.
    """

    # observation config modifiers for these experiments
    def change_lr(config):
        with config.algo.values_unlocked():
            config.algo.optim_params.policy.learning_rate.initial = 1e-3
        return config

    def change_gmm(config):
        with config.algo.values_unlocked():
            config.algo.gmm.enabled = False
        return config

    def change_mlp(config):
        with config.algo.values_unlocked():
            config.algo.actor_layer_dims = (1024, 1024) 
        return config

    def change_conv(config):
        with config.observation.values_unlocked():
            config.observation.encoder.visual_core = 'ShallowConv'
            config.observation.encoder.visual_core_kwargs = Config()
        return config

    def change_rnnd_low_dim(config):
        with config.algo.values_unlocked():
            config.algo.rnn.hidden_dim = 100
        return config

    def change_rnnd_image(config):
        with config.algo.values_unlocked():
            config.algo.rnn.hidden_dim = 400
        return config

    hyper_ablation_json_paths = Config() # use for convenient nested dict
    for task in ["square", "transport"]:
        for dataset_type in ["ph", "mh"]:
            for hdf5_type in ["low_dim", "image"]:

                # observation modifiers to apply
                if hdf5_type == "low_dim":
                    hyper_modifiers = [change_lr, change_gmm, change_mlp, change_rnnd_low_dim]
                else:
                    hyper_modifiers = [change_lr, change_gmm, change_conv, change_rnnd_image]

                # only bc and bc-rnn
                algo_name = "bc_rnn"
                for hyper_modifier in hyper_modifiers:
                    # generate config for this experiment
                    config, json_path = generate_experiment_config(
                        base_exp_name="hyper_ablation",
                        base_config_dir=base_config_dir,
                        base_dataset_dir=base_dataset_dir,
                        base_output_dir=base_output_dir,
                        algo_name=algo_name, 
                        algo_config_modifier=algo_to_config_modifier[algo_name], 
                        task_name=task, 
                        dataset_type=dataset_type, 
                        hdf5_type=hdf5_type,
                        additional_name=hyper_modifier.__name__,
                        additional_config_modifier=hyper_modifier,
                    )

                    # save json path into dict
                    algo_name_str = "{}_{}".format(algo_name, hyper_modifier.__name__)
                    hyper_ablation_json_paths[task][dataset_type][hdf5_type][algo_name_str] = json_path

    return hyper_ablation_json_paths


def generate_d4rl_configs(
    base_config_dir, 
    base_dataset_dir, 
    base_output_dir, 
    algo_to_config_modifier, 
):
    """
    Helper function to generate all configs for reproducing BCQ, CQL, and TD3-BC runs on some D4RL
    environments.

    Args:
        base_config_dir (str): base directory to place generated configs

        base_dataset_dir (str): path to directory where datasets are on disk.
            Directory structure is expected to be consistent with the output
            of @make_dataset_dirs in the download_datasets.py script.

        base_output_dir (str): directory to save training results to. If None, will use the directory
            from the default algorithm configs.

        algo_to_config_modifier (dict): dictionary that maps algo name to a function that modifies configs 
            to add algo hyperparameter settings, given the task, dataset, and hdf5 types.
    """

    def bcq_algo_config_modifier(config):
        with config.algo.values_unlocked():
            # all LRs 1e-3, enable actor
            config.algo.optim_params.critic.learning_rate.initial = 1e-3
            config.algo.optim_params.action_sampler.learning_rate.initial = 1e-3
            config.algo.optim_params.actor.learning_rate.initial = 1e-3
            config.algo.actor.enabled = True
            config.algo.action_sampler.vae.kl_weight = 0.5
        return config

    def cql_algo_config_modifier(config):
        with config.algo.values_unlocked():
            # taken from TD3-BC settings describe in their paper
            config.algo.optim_params.critic.learning_rate.initial = 3e-4
            config.algo.optim_params.actor.learning_rate.initial = 3e-5
            config.algo.actor.bc_start_steps = 40000                        # pre-training steps for actor
            config.algo.critic.target_q_gap = None                          # no Lagrange, and fixed weight of 10.0
            config.algo.critic.cql_weight = 10.0
            config.algo.critic.min_q_weight = 1.0 
            config.algo.critic.deterministic_backup = True                  # deterministic backup (no entropy in Q-target)
            config.algo.actor.layer_dims = (256, 256, 256)                  # MLP sizes
            config.algo.critic.layer_dims = (256, 256, 256)
        return config

    d4rl_tasks = [
        # "halfcheetah-random-v0",
        # "hopper-random-v0",
        # "walker2d-random-v0",
        "halfcheetah-medium-v0",
        "hopper-medium-v0",
        "walker2d-medium-v0",
        "halfcheetah-expert-v0",
        "hopper-expert-v0",
        "walker2d-expert-v0",
        "halfcheetah-medium-expert-v0",
        "hopper-medium-expert-v0",
        "walker2d-medium-expert-v0",
        # "halfcheetah-medium-replay-v0",
        # "hopper-medium-replay-v0",
        # "walker2d-medium-replay-v0",
    ]
    d4rl_json_paths = Config() # use for convenient nested dict
    for task_name in d4rl_tasks:
        for algo_name in ["bcq", "cql", "td3_bc"]:
            config = config_factory(algo_name=algo_name)

            # hack: copy experiment and train sections from td3-bc, since that has defaults for training with D4RL
            if algo_name != "td3_bc":
                ref_config = config_factory(algo_name="td3_bc")
                with config.values_unlocked():
                    config.experiment = ref_config.experiment
                    config.train = ref_config.train
                    config.observation = ref_config.observation
                    config.train.hdf5_normalize_obs = False # only TD3-BC uses observation normalization

            # modify algo section for d4rl defaults
            if algo_name == "bcq":
                config = bcq_algo_config_modifier(config)
            elif algo_name == "cql":
                config = cql_algo_config_modifier(config)

            # set experiment name
            with config.experiment.values_unlocked():
                config.experiment.name = "{}_{}_{}".format("d4rl", algo_name, task_name)
            # set output folder and dataset
            with config.train.values_unlocked():
                if base_output_dir is None:
                    base_output_dir = "../{}_trained_models".format(algo_name)
                config.train.output_dir = os.path.join(base_output_dir, "d4rl", algo_name, task_name, "trained_models")
                config.train.data = os.path.join(base_dataset_dir, "d4rl", "converted", 
                    "{}.hdf5".format(task_name.replace("-", "_")))

            # save config to json file
            dir_to_save = os.path.join(base_config_dir, "d4rl", task_name)
            os.makedirs(dir_to_save, exist_ok=True)
            json_path = os.path.join(dir_to_save, "{}.json".format(algo_name))
            config.dump(filename=json_path)

            # save json path into dict
            d4rl_json_paths[task_name][""][""][algo_name] = json_path

    return d4rl_json_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Directory where generated configs will be placed
    parser.add_argument(
        "--config_dir",
        type=str,
        default=None,
        help="Directory where generated configs will be placed. Defaults to 'paper' subfolder in exps folder of repository",
    )

    # directory where released datasets are located
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=None,
        help="Base dataset directory for released datasets. Defaults to datasets folder in repository.",
    )

    # output directory for training runs (will be written to configs)
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Base output directory for all training runs that will be written to generated configs.",
    )

    args = parser.parse_args()

    # read args
    generated_configs_base_dir = args.config_dir
    if generated_configs_base_dir is None:
        generated_configs_base_dir = os.path.join(robomimic.__path__[0], "exps/paper")

    datasets_base_dir = args.dataset_dir
    if datasets_base_dir is None:
        datasets_base_dir = os.path.join(robomimic.__path__[0], "../datasets")

    output_base_dir = args.output_dir

    # algo to modifier
    algo_to_modifier = dict(
        bc=modify_bc_config_for_dataset, 
        bc_rnn=modify_bc_rnn_config_for_dataset,
        bcq=modify_bcq_config_for_dataset,
        cql=modify_cql_config_for_dataset,
        hbc=modify_hbc_config_for_dataset,
        iris=modify_iris_config_for_dataset,
    )

    # exp name to config generator
    exp_name_to_generator = dict(
        core=generate_core_configs,
        subopt=generate_subopt_configs,
        dataset_size=generate_dataset_size_configs,
        obs_ablation=generate_obs_ablation_configs,
        hyper_ablation=generate_hyper_ablation_configs,
        d4rl=generate_d4rl_configs,
    )

    # generate configs for each experiment name
    config_json_paths = Config() # use for convenient nested dict
    for exp_name in exp_name_to_generator:
        config_json_paths[exp_name] = exp_name_to_generator[exp_name](
            base_config_dir=generated_configs_base_dir, 
            base_dataset_dir=datasets_base_dir, 
            base_output_dir=output_base_dir, 
            algo_to_config_modifier=algo_to_modifier, 
        )

    # write output shell scripts
    for exp_name in config_json_paths:
        shell_path = os.path.join(generated_configs_base_dir, "{}.sh".format(exp_name))
        with open(shell_path, "w") as f:
            f.write("#!/bin/bash\n\n")
            f.write("# " + "=" * 10 + exp_name + "=" * 10 + "\n")
            train_script_loc = os.path.join(robomimic.__path__[0], "scripts/train.py")

            for task in config_json_paths[exp_name]:
                for dataset_type in config_json_paths[exp_name][task]:
                    for hdf5_type in config_json_paths[exp_name][task][dataset_type]:
                        f.write("\n")
                        f.write("#  task: {}\n".format(task))
                        if len(dataset_type) > 0:
                            f.write("#    dataset type: {}\n".format(dataset_type))
                        if len(hdf5_type) > 0:
                            f.write("#      hdf5 type: {}\n".format(hdf5_type))
                        for algo_name in config_json_paths[exp_name][task][dataset_type][hdf5_type]:
                            # f.write("#        {}\n".format(algo_name))
                            exp_json_path = config_json_paths[exp_name][task][dataset_type][hdf5_type][algo_name]
                            cmd = "python {} --config {}\n".format(train_script_loc, exp_json_path)
                            f.write(cmd)
            f.write("\n")
