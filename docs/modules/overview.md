# Overview

![overview](../images/module_overview.png)

The **robomimic** framework consists of several modular pieces that interact to train and evaluate a policy. A [Config](./configs.html) object is used to define all settings for a particular training run, including the hdf5 dataset that will be used to train the agent, and algorithm hyperparameters. The demonstrations in the hdf5 dataset are loaded into a [SequenceDataset](./dataset.html) object, which is used to provide minibatches for the train loop. Training consists of an [Algorithm](./algorithms.html) object that trains a set of [Models](./models.html) (including the Policy) by repeatedly sampling minibatches from the fixed, offline dataset. Every so often, the policy is evaluated in the [Environment](./environments.html) by conducting a set of rollouts. Statistics and other important information during the training process are logged to disk (e.g. tensorboard outputs, model checkpoints, and evaluation rollout videos). We also provide additional utilities in [TensorUtils](./tensor_utils.html) to work with complex observations in the form of nested tensor dictionaries.

The directory structure of the repository is as follows.

- `robomimic/algo`: policy learning algorithm implementations (see [Algorithm documentation](./algorithms.html) for more information)
- `robomimic/config`: config classes (see [Config documentation](./configs.html) for more information)
- `robomimic/envs`: wrappers for environments, used during evaluation rollouts (see [Environment documentation](./environments.html) for more information)
- `robomimic/exps/templates`: config templates for each policy learning algorithm (these are auto-generated with the `robomimic/scripts/generate_config_templates.py` script)
- `robomimic/models`: network implementations (see [Models documentation](./models.html) for more information)
- `robomimic/scripts`: main repository scripts 
- `robomimic/utils`: a collection of utilities, including the [SequenceDataset](./dataset.html) class to load hdf5 datasets into a torch training pipeline, and [TensorUtils](./tensor_utils.html) to work with nested tensor dictionaries 
- `tests`: test scripts for validating repository functionality
- `examples`: some simple examples to better understand modular components in the codebase (see the [Examples documentation](../introduction/examples.html) for more information)
- `docs`: files to generate sphinx documentation

