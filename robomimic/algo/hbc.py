"""
Implementation of Hierarchical Behavioral Cloning, where
a planner model outputs subgoals (future observations), and
an actor model is conditioned on the subgoals to try and
reach them. Largely based on the Generalization Through Imitation (GTI)
paper (see https://arxiv.org/abs/2003.06085).
"""

import textwrap
import numpy as np
from collections import OrderedDict
from copy import deepcopy

import torch

import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic.config.config import Config
from robomimic.algo import (
    register_algo_factory_func,
    algo_name_to_factory_func,
    HierarchicalAlgo,
    GL_VAE,
)


@register_algo_factory_func("hbc")
def algo_config_to_class(algo_config):
    """
    Maps algo config to the HBC algo class to instantiate, along with additional algo kwargs.

    Args:
        algo_config (Config instance): algo config

    Returns:
        algo_class: subclass of Algo
        algo_kwargs (dict): dictionary of additional kwargs to pass to algorithm
    """
    pol_cls, _ = algo_name_to_factory_func("bc")(algo_config.actor)
    plan_cls, _ = algo_name_to_factory_func("gl")(algo_config.planner)
    return HBC, dict(policy_algo_class=pol_cls, planner_algo_class=plan_cls)


class HBC(HierarchicalAlgo):
    """
    Default HBC training, largely based on https://arxiv.org/abs/2003.06085
    """

    def __init__(
        self,
        planner_algo_class,
        policy_algo_class,
        algo_config,
        obs_config,
        global_config,
        obs_key_shapes,
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

            obs_key_shapes (dict): dictionary that maps input/output observation keys to shapes

            ac_dim (int): action dimension

            device: torch device
        """
        self.algo_config = algo_config
        self.obs_config = obs_config
        self.global_config = global_config

        self.ac_dim = ac_dim
        self.device = device

        self._subgoal_step_count = (
            0  # current step count for deciding when to update subgoal
        )
        self._current_subgoal = None  # latest subgoal
        self._subgoal_update_interval = (
            self.algo_config.subgoal_update_interval
        )  # subgoal update frequency
        self._subgoal_horizon = self.algo_config.planner.subgoal_horizon
        self._actor_horizon = self.algo_config.actor.rnn.horizon

        self._algo_mode = self.algo_config.mode
        assert self._algo_mode in ["separate", "cascade"]

        self.planner = planner_algo_class(
            algo_config=algo_config.planner,
            obs_config=obs_config.planner,
            global_config=global_config,
            obs_key_shapes=obs_key_shapes,
            ac_dim=ac_dim,
            device=device,
        )

        # goal-conditional actor follows goals set by the planner
        self.actor_goal_shapes = self.planner.subgoal_shapes
        if self.algo_config.latent_subgoal.enabled:
            assert planner_algo_class == GL_VAE  # only VAE supported for now
            self.actor_goal_shapes = OrderedDict(
                latent_subgoal=(self.planner.algo_config.vae.latent_dim,)
            )

        # only for the actor: override goal modalities and shapes to match the subgoal set by the planner
        actor_obs_key_shapes = deepcopy(obs_key_shapes)
        # make sure we are not modifying existing observation key shapes
        for k in self.actor_goal_shapes:
            if k in actor_obs_key_shapes:
                assert actor_obs_key_shapes[k] == self.actor_goal_shapes[k]
        actor_obs_key_shapes.update(self.actor_goal_shapes)

        goal_obs_keys = {
            obs_modality: [] for obs_modality in ObsUtils.OBS_MODALITY_CLASSES.keys()
        }
        for k in self.actor_goal_shapes.keys():
            goal_obs_keys[ObsUtils.OBS_KEYS_TO_MODALITIES[k]].append(k)

        actor_obs_config = deepcopy(obs_config.actor)
        with actor_obs_config.unlocked():
            actor_obs_config["goal"] = Config(**goal_obs_keys)

        self.actor = policy_algo_class(
            algo_config=algo_config.actor,
            obs_config=actor_obs_config,
            global_config=global_config,
            obs_key_shapes=actor_obs_key_shapes,
            ac_dim=ac_dim,
            device=device,
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
                low=0,
                high=self.global_config.train.seq_length,
                size=(batch["actions"].shape[0],),
            )
            goal_obs = TensorUtils.gather_sequence(
                batch["next_obs"], policy_subgoal_indices
            )
            goal_obs = TensorUtils.to_float(
                TensorUtils.to_device(goal_obs, self.device)
            )
            input_batch["actor"]["goal_obs"] = (
                self.planner.get_actor_goal_for_training_from_processed_batch(
                    goal_obs,
                    use_latent_subgoals=self.algo_config.latent_subgoal.enabled,
                    use_prior_correction=self.algo_config.latent_subgoal.prior_correction.enabled,
                    num_prior_samples=self.algo_config.latent_subgoal.prior_correction.num_samples,
                )
            )
        else:
            # otherwise, use planner subgoal target as goal for the policy
            input_batch["actor"]["goal_obs"] = (
                self.planner.get_actor_goal_for_training_from_processed_batch(
                    input_batch["planner"],
                    use_latent_subgoals=self.algo_config.latent_subgoal.enabled,
                    use_prior_correction=self.algo_config.latent_subgoal.prior_correction.enabled,
                    num_prior_samples=self.algo_config.latent_subgoal.prior_correction.num_samples,
                )
            )

        # we move to device first before float conversion because image observation modalities will be uint8 -
        # this minimizes the amount of data transferred to GPU
        return TensorUtils.to_float(TensorUtils.to_device(input_batch, self.device))

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
        info = dict(planner=dict(), actor=dict())
        # train planner
        info["planner"].update(
            self.planner.train_on_batch(batch["planner"], epoch, validate=validate)
        )

        # train actor
        if self._algo_mode == "separate":
            # train low-level actor by getting subgoals from the dataset
            info["actor"].update(
                self.actor.train_on_batch(batch["actor"], epoch, validate=validate)
            )

        elif self._algo_mode == "cascade":
            # get predictions from the planner
            with torch.no_grad():
                batch["actor"]["goal_obs"] = self.planner.get_subgoal_predictions(
                    obs_dict=batch["planner"]["obs"],
                    goal_dict=batch["planner"]["goal_obs"],
                )

            # train actor with the predicted goal
            info["actor"].update(
                self.actor.train_on_batch(batch["actor"], epoch, validate=validate)
            )

        else:
            raise NotImplementedError(
                "algo mode {} is not implemented".format(self._algo_mode)
            )

        return info

    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.

        Args:
            info (dict): dictionary of info

        Returns:
            loss_log (dict): name -> summary statistic
        """
        planner_log = dict()
        actor_log = dict()
        loss = 0.0

        planner_log = self.planner.log_info(info["planner"])
        planner_log = dict(("Planner/" + k, v) for k, v in planner_log.items())
        loss += planner_log["Planner/Loss"]

        actor_log = self.actor.log_info(info["actor"])
        actor_log = dict(("Actor/" + k, v) for k, v in actor_log.items())
        loss += actor_log["Actor/Loss"]

        planner_log.update(actor_log)
        planner_log["Loss"] = loss
        return planner_log

    def on_epoch_end(self, epoch):
        """
        Called at the end of each epoch.
        """
        self.planner.on_epoch_end(epoch)
        self.actor.on_epoch_end(epoch)

    def set_eval(self):
        """
        Prepare networks for evaluation.
        """
        self.planner.set_eval()
        self.actor.set_eval()

    def set_train(self):
        """
        Prepare networks for training.
        """
        self.planner.set_train()
        self.actor.set_train()

    def serialize(self):
        """
        Get dictionary of current model parameters.
        """
        return dict(
            planner=self.planner.serialize(),
            actor=self.actor.serialize(),
        )

    def deserialize(self, model_dict):
        """
        Load model from a checkpoint.

        Args:
            model_dict (dict): a dictionary saved by self.serialize() that contains
                the same keys as @self.network_classes
        """
        self.actor.deserialize(model_dict["actor"])
        self.planner.deserialize(model_dict["planner"])

    @property
    def current_subgoal(self):
        """
        Return the current subgoal (at rollout time) with shape (batch, ...)
        """
        return {k: self._current_subgoal[k].clone() for k in self._current_subgoal}

    @current_subgoal.setter
    def current_subgoal(self, sg):
        """
        Sets the current subgoal being used by the actor.
        """
        for k, v in sg.items():
            if not self.algo_config.latent_subgoal.enabled:
                # subgoal should only match subgoal shapes if not using latent subgoals
                assert list(v.shape[1:]) == list(self.planner.subgoal_shapes[k])
            # subgoal shapes should always match actor goal shapes
            assert list(v.shape[1:]) == list(self.actor_goal_shapes[k])
        self._current_subgoal = {k: sg[k].clone() for k in sg}

    def get_action(self, obs_dict, goal_dict=None):
        """
        Get policy action outputs.

        Args:
            obs_dict (dict): current observation
            goal_dict (dict): (optional) goal

        Returns:
            action (torch.Tensor): action tensor
        """
        if (
            self._current_subgoal is None
            or self._subgoal_step_count % self._subgoal_update_interval == 0
        ):
            # update current subgoal
            self.current_subgoal = self.planner.get_subgoal_predictions(
                obs_dict=obs_dict, goal_dict=goal_dict
            )

        action = self.actor.get_action(
            obs_dict=obs_dict, goal_dict=self.current_subgoal
        )
        self._subgoal_step_count += 1
        return action

    def reset(self):
        """
        Reset algo state to prepare for environment rollouts.
        """
        self._current_subgoal = None
        self._subgoal_step_count = 0
        self.planner.reset()
        self.actor.reset()

    def __repr__(self):
        """
        Pretty print algorithm and network description.
        """
        msg = str(self.__class__.__name__)
        msg += (
            "(subgoal_horizon={}, actor_horizon={}, subgoal_update_interval={}, mode={}, "
            "actor_use_random_subgoals={})\n".format(
                self._subgoal_horizon,
                self._actor_horizon,
                self._subgoal_update_interval,
                self._algo_mode,
                self.algo_config.actor_use_random_subgoals,
            )
        )
        return (
            msg
            + "Planner:\n"
            + textwrap.indent(self.planner.__repr__(), "  ")
            + "\n\nPolicy:\n"
            + textwrap.indent(self.actor.__repr__(), "  ")
        )
