"""
This file contains base classes that other algorithm classes subclass.
Each algorithm file also implements a algorithm factory function that
takes in an algorithm config (`config.algo`) and returns the particular
Algo subclass that should be instantiated, along with any extra kwargs.
These factory functions are registered into a global dictionary with the
@register_algo_factory_func function decorator. This makes it easy for
@algo_factory to instantiate the correct `Algo` subclass.
"""
import textwrap
from copy import deepcopy
from collections import OrderedDict

import torch.nn as nn

import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils


# mapping from algo name to factory functions that map algo configs to algo class names
REGISTERED_ALGO_FACTORY_FUNCS = OrderedDict()


def register_algo_factory_func(algo_name):
    """
    Function decorator to register algo factory functions that map algo configs to algo class names.
    Each algorithm implements such a function, and decorates it with this decorator.

    Args:
        algo_name (str): the algorithm name to register the algorithm under
    """
    def decorator(factory_func):
        REGISTERED_ALGO_FACTORY_FUNCS[algo_name] = factory_func
    return decorator


def algo_name_to_factory_func(algo_name):
    """
    Uses registry to retrieve algo factory function from algo name.

    Args:
        algo_name (str): the algorithm name
    """
    return REGISTERED_ALGO_FACTORY_FUNCS[algo_name]


def algo_factory(algo_name, config, modality_shapes, ac_dim, device):
    """
    Factory function for creating algorithms based on the algorithm name and config.

    Args:
        algo_name (str): the algorithm name

        config (BaseConfig instance): config object

        modality_shapes (OrderedDict): dictionary that maps modality keys to shapes

        ac_dim (int): dimension of action space

        device (torch.Device): where the algo should live (i.e. cpu, gpu)
    """

    # @algo_name is included as an arg to be explicit, but make sure it matches the config
    assert algo_name == config.algo_name

    # use algo factory func to get algo class and kwargs from algo config
    factory_func = algo_name_to_factory_func(algo_name)
    algo_cls, algo_kwargs = factory_func(config.algo)

    # create algo instance
    return algo_cls(
        algo_config=config.algo,
        obs_config=config.observation,
        global_config=config,
        modality_shapes=modality_shapes,
        ac_dim=ac_dim,
        device=device,
        **algo_kwargs
    )


class Algo(object):
    """
    Base algorithm class that all other algorithms subclass. Defines several
    functions that should be overriden by subclasses, in order to provide
    a standard API to be used by training functions such as @run_epoch in
    utils/train_utils.py.
    """
    def __init__(
        self,
        algo_config,
        obs_config,
        global_config,
        modality_shapes,
        ac_dim,
        device
    ):
        """
        Args:
            algo_config (Config object): instance of Config corresponding to the algo section
                of the config

            obs_config (Config object): instance of Config corresponding to the observation
                section of the config

            global_config (Config object): global training config

            modality_shapes (OrderedDict): dictionary that maps modality keys to shapes

            ac_dim (int): dimension of action space

            device (torch.Device): where the algo should live (i.e. cpu, gpu)
        """
        self.optim_params = deepcopy(algo_config.optim_params)
        self.algo_config = algo_config
        self.obs_config = obs_config
        self.global_config = global_config

        self.ac_dim = ac_dim
        self.device = device
        self.modality_shapes = modality_shapes

        self.nets = nn.ModuleDict()
        self._create_shapes(obs_config.modalities, modality_shapes)
        self._create_networks()
        self._create_optimizers()
        assert isinstance(self.nets, nn.ModuleDict)

    def _create_shapes(self, modalities, modality_shapes):
        """
        Create obs_shapes, goal_shapes, and subgoal_shapes dictionaries, to make it
        easy for this algorithm object to keep track of modality shapes. Each dictionary
        maps modality to shape.

        Args:
            modalities (dict): dict of required modalities for this training run (usually 
                specified by the obs config), e.g., {"obs": ["image", "proprio"], "goal": ["proprio"]}
            modality_shapes (dict): dict of modality shapes, e.g., {"image": [3, 224, 224]}
        """
        # determine shapes
        self.obs_shapes = OrderedDict()
        self.goal_shapes = OrderedDict()
        self.subgoal_shapes = OrderedDict()
        for k in modality_shapes:
            if "obs" in self.obs_config.modalities and k in (self.obs_config.modalities.obs.low_dim + self.obs_config.modalities.obs.image):
                self.obs_shapes[k] = modality_shapes[k]
            if "goal" in self.obs_config.modalities and k in (self.obs_config.modalities.goal.low_dim + self.obs_config.modalities.goal.image):
                self.goal_shapes[k] = modality_shapes[k]
            if "subgoal" in self.obs_config.modalities and k in (self.obs_config.modalities.subgoal.low_dim + self.obs_config.modalities.subgoal.image):
                self.subgoal_shapes[k] = modality_shapes[k]

    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        @self.nets should be a ModuleDict.
        """
        raise NotImplementedError

    def _create_optimizers(self):
        """
        Creates optimizers using @self.optim_params and places them into @self.optimizers.
        """
        self.optimizers = dict()
        self.lr_schedulers = dict()

        for k in self.optim_params:
            # only make optimizers for networks that have been created - @optim_params may have more
            # settings for unused networks
            if k in self.nets:
                if isinstance(self.nets[k], nn.ModuleList):
                    self.optimizers[k] = [
                        TorchUtils.optimizer_from_optim_params(net_optim_params=self.optim_params[k], net=self.nets[k][i])
                        for i in range(len(self.nets[k]))
                    ]
                    self.lr_schedulers[k] = [
                        TorchUtils.lr_scheduler_from_optim_params(net_optim_params=self.optim_params[k], net=self.nets[k][i], optimizer=self.optimizers[k][i])
                        for i in range(len(self.nets[k]))
                    ]
                else:
                    self.optimizers[k] = TorchUtils.optimizer_from_optim_params(
                        net_optim_params=self.optim_params[k], net=self.nets[k])
                    self.lr_schedulers[k] = TorchUtils.lr_scheduler_from_optim_params(
                        net_optim_params=self.optim_params[k], net=self.nets[k], optimizer=self.optimizers[k])

    def process_batch_for_training(self, batch):
        """
        Processes input batch from a data loader to filter out
        relevant information and prepare the batch for training.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader

        Returns:
            input_batch (dict): processed and filtered batch that
                will be used for training 
        """
        return batch

    def train_on_batch(self, batch, epoch, validate=False):
        """
        Training on a single batch of data.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

            epoch (int): epoch number - required by some Algos that need
                to perform staged training and early stopping

            validate (bool): if True, don't perform any learning updates.

        Returns:
            info (dict): dictionary of relevant inputs, outputs, and losses
                that might be relevant for logging
        """
        assert validate or self.nets.training
        return OrderedDict()

    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.

        Args:
            info (dict): dictionary of info

        Returns:
            loss log (dict): name -> summary statistic
        """
        log = OrderedDict()

        # record current optimizer learning rates
        for k in self.optimizers:
            for i, param_group in enumerate(self.optimizers[k].param_groups):
                log["Optimizer/{}{}_lr".format(k, i)] = param_group["lr"]

        return log

    def on_epoch_end(self, epoch):
        """
        Called at the end of each epoch.
        """

        # LR scheduling updates
        for k in self.lr_schedulers:
            if self.lr_schedulers[k] is not None:
                self.lr_schedulers[k].step()

    def set_eval(self):
        """
        Prepare networks for evaluation.
        """
        self.nets.eval()

    def set_train(self):
        """
        Prepare networks for training.
        """
        self.nets.train()

    def serialize(self):
        """
        Get dictionary of current model parameters.
        """
        return self.nets.state_dict()

    def deserialize(self, model_dict):
        """
        Load model from a checkpoint.

        Args:
            model_dict (dict): a dictionary saved by self.serialize() that contains
                the same keys as @self.network_classes
        """
        self.nets.load_state_dict(model_dict)

    def __repr__(self):
        """
        Pretty print algorithm and network description.
        """
        return "{} (\n".format(self.__class__.__name__) + \
               textwrap.indent(self.nets.__repr__(), '  ') + "\n)"

    def reset(self):
        """
        Reset algo state to prepare for environment rollouts.
        """
        pass


class PolicyAlgo(Algo):
    """
    Base class for all algorithms that can be used as policies.
    """
    def get_action(self, obs_dict, goal_dict=None):
        """
        Get policy action outputs.

        Args:
            obs_dict (dict): current observation
            goal_dict (dict): (optional) goal

        Returns:
            action (torch.Tensor): action tensor
        """
        raise NotImplementedError


class ValueAlgo(Algo):
    """
    Base class for all algorithms that can learn a value function.
    """
    def get_state_value(self, obs_dict, goal_dict=None):
        """
        Get state value outputs.

        Args:
            obs_dict (dict): current observation
            goal_dict (dict): (optional) goal

        Returns:
            value (torch.Tensor): value tensor
        """
        raise NotImplementedError

    def get_state_action_value(self, obs_dict, actions, goal_dict=None):
        """
        Get state-action value outputs.

        Args:
            obs_dict (dict): current observation
            actions (torch.Tensor): action
            goal_dict (dict): (optional) goal

        Returns:
            value (torch.Tensor): value tensor
        """
        raise NotImplementedError


class PlannerAlgo(Algo):
    """
    Base class for all algorithms that can be used for planning subgoals
    conditioned on current observations and potential goal observations.
    """
    def get_subgoal_predictions(self, obs_dict, goal_dict=None):
        """
        Get predicted subgoal outputs.

        Args:
            obs_dict (dict): current observation
            goal_dict (dict): (optional) goal

        Returns:
            subgoal prediction (dict): name -> Tensor [batch_size, ...]
        """
        raise NotImplementedError

    def sample_subgoals(self, obs_dict, goal_dict, num_samples=1):
        """
        For planners that rely on sampling subgoals.

        Args:
            obs_dict (dict): current observation
            goal_dict (dict): (optional) goal

        Returns:
            subgoals (dict): name -> Tensor [batch_size, num_samples, ...]
        """
        raise NotImplementedError


class HierarchicalAlgo(Algo):
    """
    Base class for all hierarchical algorithms that consist of (1) subgoal planning
    and (2) subgoal-conditioned policy learning.
    """
    def get_action(self, obs_dict, goal_dict=None):
        """
        Get policy action outputs.

        Args:
            obs_dict (dict): current observation
            goal_dict (dict): (optional) goal

        Returns:
            action (torch.Tensor): action tensor
        """
        raise NotImplementedError

    def get_subgoal_predictions(self, obs_dict, goal_dict=None):
        """
        Get subgoal predictions from high-level subgoal planner.

        Args:
            obs_dict (dict): current observation
            goal_dict (dict): (optional) goal

        Returns:
            subgoal (dict): predicted subgoal
        """
        raise NotImplementedError

    @property
    def current_subgoal(self):
        """
        Get the current subgoal for conditioning the low-level policy

        Returns:
            current subgoal (dict): predicted subgoal
        """
        raise NotImplementedError


class RolloutPolicy(object):
    """
    Wraps @Algo object to make it easy to run policies in a rollout loop.
    """
    def __init__(self, policy, obs_normalization_stats=None):
        """
        Args:
            policy (Algo instance): @Algo object to wrap to prepare for rollouts

            obs_normalization_stats (dict): optionally pass a dictionary for observation
                normalization. This should map observation modality keys to dicts
                with a "mean" and "std" of shape (1, ...) where ... is the default
                shape for the observation.
        """
        self.policy = policy
        self.obs_normalization_stats = obs_normalization_stats

    def start_episode(self):
        """
        Prepare the policy to start a new rollout.
        """
        self.policy.set_eval()
        self.policy.reset()

    def _prepare_observation(self, ob):
        """
        Prepare raw observation dict from environment for policy.

        Args:
            ob (dict): single observation dictionary from environment (no batch dimension, 
                and np.array values for each key)
        """
        if self.obs_normalization_stats is not None:
            ob = ObsUtils.normalize_obs(ob, obs_normalization_stats=self.obs_normalization_stats)
        ob = TensorUtils.to_tensor(ob)
        ob = TensorUtils.to_batch(ob)
        ob = TensorUtils.to_device(ob, self.policy.device)
        ob = TensorUtils.to_float(ob)
        return ob

    def __repr__(self):
        """Pretty print network description"""
        return self.policy.__repr__()

    def __call__(self, ob, goal=None):
        """
        Produce action from raw observation dict (and maybe goal dict) from environment.

        Args:
            ob (dict): single observation dictionary from environment (no batch dimension, 
                and np.array values for each key)
            goal (dict): goal observation
        """
        ob = self._prepare_observation(ob)
        if goal is not None:
            goal = self._prepare_observation(goal)
        ac = self.policy.get_action(obs_dict=ob, goal_dict=goal)
        return TensorUtils.to_numpy(ac[0])
