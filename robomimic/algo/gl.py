"""
Subgoal prediction models, used in HBC / IRIS.
"""
import numpy as np
from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn

import robomimic.models.obs_nets as ObsNets
import robomimic.models.vae_nets as VAENets
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils

from robomimic.algo import register_algo_factory_func, PlannerAlgo, ValueAlgo


@register_algo_factory_func("gl")
def algo_config_to_class(algo_config):
    """
    Maps algo config to the GL algo class to instantiate, along with additional algo kwargs.

    Args:
        algo_config (Config instance): algo config

    Returns:
        algo_class: subclass of Algo
        algo_kwargs (dict): dictionary of additional kwargs to pass to algorithm
    """
    if algo_config.vae.enabled:
        return GL_VAE, {}
    return GL, {}


class GL(PlannerAlgo):
    """
    Implements goal prediction component for HBC and IRIS.
    """
    def __init__(
        self,
        algo_config,
        obs_config,
        global_config,
        obs_key_shapes,
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

            obs_key_shapes (OrderedDict): dictionary that maps observation keys to shapes

            ac_dim (int): dimension of action space

            device (torch.Device): where the algo should live (i.e. cpu, gpu)
        """

        self._subgoal_horizon = algo_config.subgoal_horizon
        super(GL, self).__init__(
            algo_config=algo_config,
            obs_config=obs_config,
            global_config=global_config,
            obs_key_shapes=obs_key_shapes,
            ac_dim=ac_dim,
            device=device
        )

    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        self.nets = nn.ModuleDict()

        obs_group_shapes = OrderedDict()
        obs_group_shapes["obs"] = OrderedDict(self.obs_shapes)
        if len(self.goal_shapes) > 0:
            obs_group_shapes["goal"] = OrderedDict(self.goal_shapes)

        # deterministic goal prediction network
        self.nets["goal_network"] = ObsNets.MIMO_MLP(
            input_obs_group_shapes=obs_group_shapes, 
            output_shapes=self.subgoal_shapes,
            layer_dims=self.algo_config.ae.planner_layer_dims,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
        )

        self.nets = self.nets.float().to(self.device)

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

        # remove temporal batches for all except scalar signals (to be compatible with model outputs)
        input_batch["obs"] = { k: batch["obs"][k][:, 0, :] for k in batch["obs"] }
        # extract multi-horizon subgoal target
        input_batch["subgoals"] = {k: batch["next_obs"][k][:, self._subgoal_horizon - 1, :] for k in batch["next_obs"]}
        input_batch["target_subgoals"] = input_batch["subgoals"]
        input_batch["goal_obs"] = batch.get("goal_obs", None) # goals may not be present

        # we move to device first before float conversion because image observation modalities will be uint8 -
        # this minimizes the amount of data transferred to GPU
        return TensorUtils.to_float(TensorUtils.to_device(input_batch, self.device))

    def get_actor_goal_for_training_from_processed_batch(self, processed_batch, **kwargs):
        """
        Retrieve subgoals from processed batch to use for training the actor. Subclasses
        can modify this function to change the subgoals.

        Args:
            processed_batch (dict): processed batch from @process_batch_for_training

        Returns:
            actor_subgoals (dict): subgoal observations to condition actor on
        """
        return processed_batch["target_subgoals"]

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
        with TorchUtils.maybe_no_grad(no_grad=validate):
            info = super(GL, self).train_on_batch(batch, epoch, validate=validate)

            # predict subgoal observations with goal network
            pred_subgoals = self.nets["goal_network"](obs=batch["obs"], goal=batch["goal_obs"])

            # compute loss as L2 error for each observation key
            losses = OrderedDict()
            target_subgoals = batch["target_subgoals"]  # targets for network prediction
            goal_loss = 0.
            for k in pred_subgoals:
                assert pred_subgoals[k].shape == target_subgoals[k].shape, "mismatch in predicted and target subgoals!"
                mode_loss = nn.MSELoss()(pred_subgoals[k], target_subgoals[k])
                goal_loss += mode_loss
                losses["goal_{}_loss".format(k)] = mode_loss
            losses["goal_loss"] = goal_loss
            info.update(TensorUtils.detach(losses))

            if not validate:
                # gradient step
                goal_grad_norms = TorchUtils.backprop_for_loss(
                    net=self.nets["goal_network"],
                    optim=self.optimizers["goal_network"],
                    loss=losses["goal_loss"],
                )
                info["goal_grad_norms"] = goal_grad_norms

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
        loss_log = super(GL, self).log_info(info)

        loss_log["Loss"] = info["goal_loss"].item()
        for k in info:
            if k.endswith("_loss"):
                loss_log[k] = info[k].item()
        if "goal_grad_norms" in info:
            loss_log["Grad_Norms"] = info["goal_grad_norms"]

        return loss_log

    def get_subgoal_predictions(self, obs_dict, goal_dict=None):
        """
        Takes a batch of observations and predicts a batch of subgoals.

        Args:
            obs_dict (dict): current observation
            goal_dict (dict): (optional) goal

        Returns:
            subgoal prediction (dict): name -> Tensor [batch_size, ...]
        """
        return self.nets["goal_network"](obs=obs_dict, goal=goal_dict)

    def sample_subgoals(self, obs_dict, goal_dict=None, num_samples=1):
        """
        Sample @num_samples subgoals from the network per observation.
        Since this class implements a deterministic subgoal prediction, 
        this function returns identical subgoals for each input observation.

        Args:
            obs_dict (dict): current observation
            goal_dict (dict): (optional) goal

        Returns:
            subgoals (dict): name -> Tensor [batch_size, num_samples, ...]
        """

        # stack observations to get all samples in one forward pass
        obs_tiled = ObsUtils.repeat_and_stack_observation(obs_dict, n=num_samples)
        goal_tiled = None
        if goal_dict is not None:
            goal_tiled = ObsUtils.repeat_and_stack_observation(goal_dict, n=num_samples)

        # [batch_size * num_samples, ...]
        goals = self.get_subgoal_predictions(obs_dict=obs_tiled, goal_dict=goal_tiled)
        # reshape to [batch_size, num_samples, ...]
        return TensorUtils.reshape_dimensions(goals, begin_axis=0, end_axis=0, target_dims=(-1, num_samples))

    def get_action(self, obs_dict, goal_dict=None):
        """
        Get policy action outputs. Assumes one input observation (first dimension should be 1).

        Args:
            obs_dict (dict): current observation
            goal_dict (dict): (optional) goal

        Returns:
            action (torch.Tensor): action tensor
        """
        raise Exception("Rollouts are not supported by GL")


class GL_VAE(GL):
    """
    Implements goal prediction via VAE.
    """
    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        self.nets = nn.ModuleDict()

        self.nets["goal_network"] = VAENets.VAE(
            input_shapes=self.subgoal_shapes,
            output_shapes=self.subgoal_shapes,
            condition_shapes=self.obs_shapes,
            goal_shapes=self.goal_shapes,
            device=self.device,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
            **VAENets.vae_args_from_config(self.algo_config.vae),
        )

        self.nets = self.nets.float().to(self.device)

    def get_actor_goal_for_training_from_processed_batch(
        self,
        processed_batch,
        use_latent_subgoals=False,
        use_prior_correction=False,
        num_prior_samples=100,
        **kwargs,
    ):
        """
        Modify from superclass to support a @use_latent_subgoals option.
        The VAE can optionally return latent subgoals by passing the subgoal 
        observations in the batch through the encoder.

        Args:
            processed_batch (dict): processed batch from @process_batch_for_training

            use_latent_subgoals (bool): if True, condition the actor on latent subgoals
                by using the VAE encoder to encode subgoal observations at train-time,
                and using the VAE prior to generate latent subgoals at test-time

            use_prior_correction (bool): if True, use a "prior correction" trick to
                choose a latent subgoal sampled from the prior that is close to the
                latent from the VAE encoder (posterior). This can help with issues at 
                test-time where the encoder latent distribution might not match 
                the prior latent distribution.

            num_prior_samples (int): number of VAE prior samples to take and choose among,
                if @use_prior_correction is true

        Returns:
            actor_subgoals (dict): subgoal observations to condition actor on
        """

        if not use_latent_subgoals:
            return processed_batch["target_subgoals"]

        # batch variables
        obs = processed_batch["obs"]
        subgoals = processed_batch["subgoals"]  # full subgoal observations
        target_subgoals = processed_batch["target_subgoals"]  # targets for network prediction
        goal_obs = processed_batch["goal_obs"]

        with torch.no_grad():
            # run VAE forward pass to get samples from posterior for the current observation and subgoal
            vae_outputs = self.nets["goal_network"](
                inputs=subgoals, # encoder takes full subgoals
                outputs=target_subgoals, # reconstruct target subgoals
                goals=goal_obs,
                conditions=obs, # condition on observations
            )
            posterior_z = vae_outputs["encoder_z"]
            latent_subgoals = posterior_z

            if use_prior_correction:
                # instead of treating posterior samples as latent subgoals, sample latents from
                # the prior and choose the closest one as the latent subgoal

                random_key = list(obs.keys())[0]
                batch_size = obs[random_key].shape[0]

                # for each batch member, get @num_prior_samples samples from the prior
                obs_tiled = ObsUtils.repeat_and_stack_observation(obs, n=num_prior_samples)
                goal_tiled = None
                if len(self.goal_shapes) > 0:
                    goal_tiled = ObsUtils.repeat_and_stack_observation(goal_obs, n=num_prior_samples)

                prior_z_samples = self.nets["goal_network"].sample_prior(
                    conditions=obs_tiled,
                    goals=goal_tiled,
                )

                # choose prior samples that are closest to the sampled posterior latents
                # note: every posterior sample in the batch has @num_prior_samples corresponding prior samples

                # reshape prior samples to (batch_size, num_samples, latent_dim)
                prior_z_samples = prior_z_samples.reshape(batch_size, num_prior_samples, -1)

                # reshape posterior latents to (batch_size, 1, latent_dim)
                posterior_z_expanded = posterior_z.unsqueeze(1)

                # compute distances with broadcasting so that each posterior sample
                # has distances to all of its prior samples
                distances = (prior_z_samples - posterior_z_expanded).pow(2).sum(dim=2)

                # then gather the closest prior sample for each posterior sample
                neighbors = torch.argmin(distances, dim=1)
                latent_subgoals = prior_z_samples[torch.arange(batch_size).long(), neighbors]

        return { "latent_subgoal" : latent_subgoals }

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
        with TorchUtils.maybe_no_grad(no_grad=validate):
            info = super(GL, self).train_on_batch(batch, epoch, validate=validate)

            if self.algo_config.vae.prior.use_categorical:
                temperature = self.algo_config.vae.prior.categorical_init_temp - epoch * self.algo_config.vae.prior.categorical_temp_anneal_step
                temperature = max(temperature, self.algo_config.vae.prior.categorical_min_temp)
                self.nets["goal_network"].set_gumbel_temperature(temperature)

            # batch variables
            obs = batch["obs"]
            subgoals = batch["subgoals"]  # full subgoal observations
            target_subgoals = batch["target_subgoals"]  # targets for network prediction
            goal_obs = batch["goal_obs"]

            vae_outputs = self.nets["goal_network"](
                inputs=subgoals, # encoder takes full subgoals
                outputs=target_subgoals, # reconstruct target subgoals
                goals=goal_obs,
                conditions=obs, # condition on observations
            )
            recons_loss = vae_outputs["reconstruction_loss"]
            kl_loss = vae_outputs["kl_loss"]
            goal_loss = recons_loss + self.algo_config.vae.kl_weight * kl_loss
            info["recons_loss"] = recons_loss
            info["kl_loss"] = kl_loss
            info["goal_loss"] = goal_loss

            if not self.algo_config.vae.prior.use_categorical:
                with torch.no_grad():
                    info["encoder_variance"] = torch.exp(vae_outputs["encoder_params"]["logvar"])

            # VAE gradient step
            if not validate:
                goal_grad_norms = TorchUtils.backprop_for_loss(
                    net=self.nets["goal_network"],
                    optim=self.optimizers["goal_network"],
                    loss=goal_loss,
                )
                info["goal_grad_norms"] = goal_grad_norms

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
        loss_log = super(GL_VAE, self).log_info(info)
        loss_log["Reconstruction_Loss"] = info["recons_loss"].item()
        loss_log["KL_Loss"] = info["kl_loss"].item()
        if self.algo_config.vae.prior.use_categorical:
            loss_log["Gumbel_Temperature"] = self.nets["goal_network"].get_gumbel_temperature()
        else:
            loss_log["Encoder_Variance"] = info["encoder_variance"].mean().item()
        return loss_log

    def get_subgoal_predictions(self, obs_dict, goal_dict=None):
        """
        Takes a batch of observations and predicts a batch of subgoals.

        Args:
            obs_dict (dict): current observation
            goal_dict (dict): (optional) goal

        Returns:
            subgoal prediction (dict): name -> Tensor [batch_size, ...]
        """

        if self.global_config.algo.latent_subgoal.enabled:
            # latent subgoals from sampling prior
            latent_subgoals = self.nets["goal_network"].sample_prior(
                conditions=obs_dict,
                goals=goal_dict,
            )

            return OrderedDict(latent_subgoal=latent_subgoals)

        # sample a single goal from the VAE
        goals = self.sample_subgoals(obs_dict=obs_dict, goal_dict=goal_dict, num_samples=1)
        return { k : goals[k][:, 0, ...] for k in goals }

    def sample_subgoals(self, obs_dict, goal_dict=None, num_samples=1):
        """
        Sample @num_samples subgoals from the VAE per observation.

        Args:
            obs_dict (dict): current observation
            goal_dict (dict): (optional) goal

        Returns:
            subgoals (dict): name -> Tensor [batch_size, num_samples, ...]
        """

        # stack observations to get all samples in one forward pass
        obs_tiled = ObsUtils.repeat_and_stack_observation(obs_dict, n=num_samples)
        goal_tiled = None
        if goal_dict is not None:
            goal_tiled = ObsUtils.repeat_and_stack_observation(goal_dict, n=num_samples)

        # VAE decode expects number of samples explicitly
        mod = list(obs_tiled.keys())[0]
        n = obs_tiled[mod].shape[0]
        # [batch_size * num_samples, ...]
        goals = self.nets["goal_network"].decode(n=n, conditions=obs_tiled, goals=goal_tiled)
        # reshape to [batch_size, num_samples, ...]
        return TensorUtils.reshape_dimensions(goals, begin_axis=0, end_axis=0, target_dims=(-1, num_samples))


class ValuePlanner(PlannerAlgo, ValueAlgo):
    """
    Base class for all algorithms that are used for planning subgoals
    based on (1) a @PlannerAlgo that is used to sample candidate subgoals
    and (2) a @ValueAlgo that is used to select one of the subgoals.
    """
    def __init__(
        self,
        planner_algo_class,
        value_algo_class,
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

            value_algo_class (Algo class): algo class for the value network

            algo_config (Config object): instance of Config corresponding to the algo section
                of the config

            obs_config (Config object): instance of Config corresponding to the observation
                section of the config

            global_config (Config object); global config

            obs_key_shapes (OrderedDict): dictionary that maps input/output observation keys to shapes

            ac_dim (int): action dimension

            device: torch device
        """
        self.algo_config = algo_config
        self.obs_config = obs_config
        self.global_config = global_config

        self.ac_dim = ac_dim
        self.device = device

        self.planner = planner_algo_class(
            algo_config=algo_config.planner,
            obs_config=obs_config.planner,
            global_config=global_config,
            obs_key_shapes=obs_key_shapes,
            ac_dim=ac_dim,
            device=device
        )

        self.value_net = value_algo_class(
            algo_config=algo_config.value,
            obs_config=obs_config.value,
            global_config=global_config,
            obs_key_shapes=obs_key_shapes,
            ac_dim=ac_dim,
            device=device
        )

        self.subgoal_shapes = self.planner.subgoal_shapes

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
        input_batch["value_net"] = self.value_net.process_batch_for_training(batch)

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
        if validate:
            assert not self.planner.nets.training
            assert not self.value_net.nets.training

        info = dict(planner=dict(), value_net=dict())

        # train planner
        info["planner"].update(self.planner.train_on_batch(batch["planner"], epoch, validate=validate))

        # train value network
        info["value_net"].update(self.value_net.train_on_batch(batch["value_net"], epoch, validate=validate))

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
        loss = 0.

        # planner
        planner_log = self.planner.log_info(info["planner"])
        planner_log = dict(("Planner/" + k, v) for k, v in planner_log.items())
        loss += planner_log["Planner/Loss"]

        # value network
        value_net_log = self.value_net.log_info(info["value_net"])
        value_net_log = dict(("ValueNetwork/" + k, v) for k, v in value_net_log.items())
        loss += value_net_log["ValueNetwork/Loss"]
        planner_log.update(value_net_log)

        planner_log["Loss"] = loss
        return planner_log

    def on_epoch_end(self, epoch):
        """
        Called at the end of each epoch.
        """
        self.planner.on_epoch_end(epoch)
        self.value_net.on_epoch_end(epoch)

    def set_eval(self):
        """
        Prepare networks for evaluation.
        """
        self.planner.set_eval()
        self.value_net.set_eval()

    def set_train(self):
        """
        Prepare networks for training.
        """
        self.planner.set_train()
        self.value_net.set_train()

    def serialize(self):
        """
        Get dictionary of current model parameters.
        """
        return dict(
            planner=self.planner.serialize(),
            value_net=self.value_net.serialize(),
        )

    def deserialize(self, model_dict):
        """
        Load model from a checkpoint.

        Args:
            model_dict (dict): a dictionary saved by self.serialize() that contains
                the same keys as @self.network_classes
        """
        self.planner.deserialize(model_dict["planner"])
        self.value_net.deserialize(model_dict["value_net"])

    def reset(self):
        """
        Reset algo state to prepare for environment rollouts.
        """
        self.planner.reset()
        self.value_net.reset()

    def __repr__(self):
        """
        Pretty print algorithm and network description.
        """
        msg = str(self.__class__.__name__)
        import textwrap
        return msg + "Planner:\n" + textwrap.indent(self.planner.__repr__(), '  ') + \
               "\n\nValue Network:\n" + textwrap.indent(self.value_net.__repr__(), '  ')

    def get_subgoal_predictions(self, obs_dict, goal_dict=None):
        """
        Takes a batch of observations and predicts a batch of subgoals.

        Args:
            obs_dict (dict): current observation
            goal_dict (dict): (optional) goal

        Returns:
            subgoal prediction (dict): name -> Tensor [batch_size, ...]
        """

        num_samples = self.algo_config.num_samples

        # sample subgoals from the planner (shape: [batch_size, num_samples, ...])
        subgoals = self.sample_subgoals(obs_dict=obs_dict, goal_dict=goal_dict, num_samples=num_samples)

        # stack subgoals to get all values in one forward pass (shape [batch_size * num_samples, ...])
        k = list(obs_dict.keys())[0]
        bsize = obs_dict[k].shape[0]
        subgoals_tiled = TensorUtils.reshape_dimensions(subgoals, begin_axis=0, end_axis=1, target_dims=(bsize * num_samples,))

        # also repeat goals if necessary
        goal_tiled = None
        if len(self.planner.goal_shapes) > 0:
            goal_tiled = ObsUtils.repeat_and_stack_observation(goal_dict, n=num_samples)

        # evaluate the value of each subgoal
        subgoal_values = self.value_net.get_state_value(obs_dict=subgoals_tiled, goal_dict=goal_tiled).reshape(-1, num_samples)

        # pick the best subgoal
        best_index = torch.argmax(subgoal_values, dim=1)
        best_subgoal = {k: subgoals[k][torch.arange(bsize), best_index] for k in subgoals}
        return best_subgoal

    def sample_subgoals(self, obs_dict, goal_dict, num_samples=1):
        """
        Sample @num_samples subgoals from the planner algo per observation.

        Args:
            obs_dict (dict): current observation
            goal_dict (dict): (optional) goal

        Returns:
            subgoals (dict): name -> Tensor [batch_size, num_samples, ...]
        """
        return self.planner.sample_subgoals(obs_dict=obs_dict, goal_dict=goal_dict, num_samples=num_samples)

    def get_state_value(self, obs_dict, goal_dict=None):
        """
        Get state value outputs.

        Args:
            obs_dict (dict): current observation
            goal_dict (dict): (optional) goal

        Returns:
            value (torch.Tensor): value tensor
        """
        return self.value_net.get_state_value(obs_dict=obs_dict, goal_dict=goal_dict)

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
        return self.value_net.get_state_action_value(obs_dict=obs_dict, actions=actions, goal_dict=goal_dict)
