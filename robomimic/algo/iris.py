"""
Implementation of IRIS (https://arxiv.org/abs/1911.05321).
"""
import numpy as np
from collections import OrderedDict
from copy import deepcopy

import torch

import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic.config.config import Config
from robomimic.algo import register_algo_factory_func, algo_name_to_factory_func, HBC, ValuePlanner, ValueAlgo, GL_VAE


@register_algo_factory_func("iris")
def algo_config_to_class(algo_config):
    """
    Maps algo config to the IRIS algo class to instantiate, along with additional algo kwargs.

    Args:
        algo_config (Config instance): algo config

    Returns:
        algo_class: subclass of Algo
        algo_kwargs (dict): dictionary of additional kwargs to pass to algorithm
    """
    pol_cls, _ = algo_name_to_factory_func("bc")(algo_config.actor)
    plan_cls, _ = algo_name_to_factory_func("gl")(algo_config.value_planner.planner)
    value_cls, _ = algo_name_to_factory_func("bcq")(algo_config.value_planner.value)
    return IRIS, dict(policy_algo_class=pol_cls, planner_algo_class=plan_cls, value_algo_class=value_cls)


class IRIS(HBC, ValueAlgo):
    """
    Implementation of IRIS (https://arxiv.org/abs/1911.05321).
    """
    def __init__(
        self,
        planner_algo_class,
        value_algo_class,
        policy_algo_class,
        algo_config,
        obs_config,
        global_config,
        modality_shapes,
        ac_dim,
        device,
    ):
        """
        Args:
            planner_algo_class (Algo class): algo class for the planner

            policy_algo_class (Algo class): algo class for the policy

            algo_config (Config object): instance of Config corresponding to the algo section
                of the config

            obs_config (Config object): instance of Config corresponding to the observation
                section of the config

            global_config (Config object): global training config

            modality_shapes (OrderedDict): dictionary that maps input/output modality keys to shapes

            ac_dim (int): action dimension

            device: torch device
        """
        self.algo_config = algo_config
        self.obs_config = obs_config
        self.global_config = global_config

        self.ac_dim = ac_dim
        self.device = device

        self._subgoal_step_count = 0  # current step count for deciding when to update subgoal
        self._current_subgoal = None  # latest subgoal
        self._subgoal_update_interval = self.algo_config.subgoal_update_interval  # subgoal update frequency
        self._subgoal_horizon = self.algo_config.value_planner.planner.subgoal_horizon
        self._actor_horizon = self.algo_config.actor.rnn.horizon

        self._algo_mode = self.algo_config.mode
        assert self._algo_mode in ["separate", "cascade"]

        self.planner = ValuePlanner(
            planner_algo_class=planner_algo_class,
            value_algo_class=value_algo_class,
            algo_config=algo_config.value_planner,
            obs_config=obs_config.value_planner,
            global_config=global_config,
            modality_shapes=modality_shapes,
            ac_dim=ac_dim,
            device=device
        )

        self.actor_goal_shapes = self.planner.subgoal_shapes
        assert not algo_config.latent_subgoal.enabled, "IRIS does not support latent subgoals"

        # only for the actor: override goal modalities and shapes to match the subgoal set by the planner
        actor_modality_shapes = deepcopy(modality_shapes)
        # make sure we are not modifying existing modality shapes
        for k in self.actor_goal_shapes:
            if k in actor_modality_shapes:
                assert actor_modality_shapes[k] == self.actor_goal_shapes[k]
        actor_modality_shapes.update(self.actor_goal_shapes)

        goal_modalities = {"low_dim": [], "image": []}
        for k in self.actor_goal_shapes.keys():
            if ObsUtils.key_is_image(k):
                goal_modalities["image"].append(k)
            else:
                goal_modalities["low_dim"].append(k)

        actor_obs_config = deepcopy(obs_config.actor)
        with actor_obs_config.unlocked():
            actor_obs_config["goal"] = Config(**goal_modalities)

        self.actor = policy_algo_class(
            algo_config=algo_config.actor,
            obs_config=actor_obs_config,
            global_config=global_config,
            modality_shapes=actor_modality_shapes,
            ac_dim=ac_dim,
            device=device
        )

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
        input_batch = dict()

        input_batch["planner"] = self.planner.process_batch_for_training(batch)
        input_batch["actor"] = self.actor.process_batch_for_training(batch)

        if self.algo_config.actor_use_random_subgoals:
            # optionally use randomly sampled step between [1, seq_length] as policy goal
            policy_subgoal_indices = torch.randint(
                low=0, high=self.global_config.train.seq_length, size=(batch["actions"].shape[0],))
            goal_obs = TensorUtils.gather_sequence(batch["next_obs"], policy_subgoal_indices)
            goal_obs = TensorUtils.to_device(TensorUtils.to_float(goal_obs), self.device)
            input_batch["actor"]["goal_obs"] = goal_obs
        else:
            # otherwise, use planner subgoal target as goal for the policy
            input_batch["actor"]["goal_obs"] = input_batch["planner"]["planner"]["target_subgoals"]

        return TensorUtils.to_device(TensorUtils.to_float(input_batch), self.device)

    def get_state_value(self, obs_dict, goal_dict=None):
        """
        Get state value outputs.

        Args:
            obs_dict (dict): current observation
            goal_dict (dict): (optional) goal

        Returns:
            value (torch.Tensor): value tensor
        """
        return self.planner.get_state_value(obs_dict=obs_dict, goal_dict=goal_dict)

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
        return self.planner.get_state_action_value(obs_dict=obs_dict, actions=actions, goal_dict=goal_dict)
