"""
Batch-Constrained Q-Learning (BCQ), with support for more general
generative action models (the original paper uses a cVAE).
(Paper - https://arxiv.org/abs/1812.02900).
"""
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

import robomimic.models.obs_nets as ObsNets
import robomimic.models.policy_nets as PolicyNets
import robomimic.models.value_nets as ValueNets
import robomimic.models.vae_nets as VAENets
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.loss_utils as LossUtils

from robomimic.algo import register_algo_factory_func, PolicyAlgo, ValueAlgo


@register_algo_factory_func("bcq")
def algo_config_to_class(algo_config):
    """
    Maps algo config to the BCQ algo class to instantiate, along with additional algo kwargs.

    Args:
        algo_config (Config instance): algo config

    Returns:
        algo_class: subclass of Algo
        algo_kwargs (dict): dictionary of additional kwargs to pass to algorithm
    """
    if algo_config.critic.distributional.enabled:
        return BCQ_Distributional, {}
    if algo_config.action_sampler.gmm.enabled:
        return BCQ_GMM, {}
    assert algo_config.action_sampler.vae.enabled
    return BCQ, {}


class BCQ(PolicyAlgo, ValueAlgo):
    """
    Default BCQ training, based on https://arxiv.org/abs/1812.02900 and
    https://github.com/sfujim/BCQ
    """
    def __init__(self, **kwargs):
        PolicyAlgo.__init__(self, **kwargs)

        # save the discount factor - it may be overriden later
        self.set_discount(self.algo_config.discount)

    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        self.nets = nn.ModuleDict()

        self._create_critics()
        self._create_action_sampler()
        if self.algo_config.actor.enabled:
            self._create_actor()

        # sync target networks at beginning of training
        with torch.no_grad():
            for critic_ind in range(len(self.nets["critic"])):
                TorchUtils.hard_update(
                    source=self.nets["critic"][critic_ind], 
                    target=self.nets["critic_target"][critic_ind],
                )

            if self.algo_config.actor.enabled:
                TorchUtils.hard_update(
                    source=self.nets["actor"], 
                    target=self.nets["actor_target"],
                )

        self.nets = self.nets.float().to(self.device)

    def _create_critics(self):
        """
        Called in @_create_networks to make critic networks.
        """
        critic_class = ValueNets.ActionValueNetwork
        critic_args = dict(
            obs_shapes=self.obs_shapes,
            ac_dim=self.ac_dim,
            mlp_layer_dims=self.algo_config.critic.layer_dims,
            value_bounds=self.algo_config.critic.value_bounds,
            goal_shapes=self.goal_shapes,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
        )

        # Q network ensemble and target ensemble
        self.nets["critic"] = nn.ModuleList()
        self.nets["critic_target"] = nn.ModuleList()
        for _ in range(self.algo_config.critic.ensemble.n):
            critic = critic_class(**critic_args)
            self.nets["critic"].append(critic)

            critic_target = critic_class(**critic_args)
            self.nets["critic_target"].append(critic_target)

    def _create_action_sampler(self):
        """
        Called in @_create_networks to make action sampler network.
        """

        # VAE network for approximate sampling from batch dataset
        assert self.algo_config.action_sampler.vae.enabled
        self.nets["action_sampler"] = PolicyNets.VAEActor(
            obs_shapes=self.obs_shapes,
            ac_dim=self.ac_dim,
            device=self.device,
            goal_shapes=self.goal_shapes,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
            **VAENets.vae_args_from_config(self.algo_config.action_sampler.vae),
        )

    def _create_actor(self):
        """
        Called in @_create_networks to make actor network.
        """
        assert self.algo_config.actor.enabled
        actor_class = PolicyNets.PerturbationActorNetwork
        actor_args = dict(
            obs_shapes=self.obs_shapes,
            goal_shapes=self.goal_shapes,
            ac_dim=self.ac_dim,
            mlp_layer_dims=self.algo_config.actor.layer_dims,
            perturbation_scale=self.algo_config.actor.perturbation_scale,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
        )

        self.nets["actor"] = actor_class(**actor_args)
        self.nets["actor_target"] = actor_class(**actor_args)

    def _check_epoch(self, net_name, epoch):
        """
        Helper function to check whether backprop should happen this epoch.

        Args:
            net_name (str): name of network in @self.nets and @self.optim_params
            epoch (int): epoch number
        """
        epoch_start_check = (self.optim_params[net_name]["start_epoch"] == -1) or (epoch >= self.optim_params[net_name]["start_epoch"])
        epoch_end_check = (self.optim_params[net_name]["end_epoch"] == -1) or (epoch < self.optim_params[net_name]["end_epoch"])
        return (epoch_start_check and epoch_end_check)

    def set_discount(self, discount):
        """
        Useful function to modify discount factor if necessary (e.g. for n-step returns).
        """
        self.discount = discount

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

        # n-step returns (default is 1)
        n_step = self.algo_config.n_step
        assert batch["actions"].shape[1] >= n_step

        # remove temporal batches for all
        input_batch["obs"] = {k: batch["obs"][k][:, 0, :] for k in batch["obs"]}
        input_batch["next_obs"] = {k: batch["next_obs"][k][:, n_step - 1, :] for k in batch["next_obs"]}
        input_batch["goal_obs"] = batch.get("goal_obs", None) # goals may not be present
        input_batch["actions"] = batch["actions"][:, 0, :]

        # note: ensure scalar signals (rewards, done) retain last dimension of 1 to be compatible with model outputs

        # single timestep reward is discounted sum of intermediate rewards in sequence
        reward_seq = batch["rewards"][:, :n_step]
        discounts = torch.pow(self.algo_config.discount, torch.arange(n_step).float()).unsqueeze(0)
        input_batch["rewards"] = (reward_seq * discounts).sum(dim=1).unsqueeze(1)

        # discount rate will be gamma^N for computing n-step returns
        new_discount = (self.algo_config.discount ** n_step)
        self.set_discount(new_discount)

        # consider this n-step seqeunce done if any intermediate dones are present
        done_seq = batch["dones"][:, :n_step]
        input_batch["dones"] = (done_seq.sum(dim=1) > 0).float().unsqueeze(1)

        if self.algo_config.infinite_horizon:
            # scale terminal rewards by 1 / (1 - gamma) for infinite horizon MDPs
            done_inds = input_batch["dones"].round().long().nonzero(as_tuple=False)[:, 0]
            if done_inds.shape[0] > 0:
                input_batch["rewards"][done_inds] = input_batch["rewards"][done_inds] * (1. / (1. - self.discount))

        # we move to device first before float conversion because image observation modalities will be uint8 -
        # this minimizes the amount of data transferred to GPU
        return TensorUtils.to_float(TensorUtils.to_device(input_batch, self.device))

    def _train_action_sampler_on_batch(self, batch, epoch, no_backprop=False):
        """
        A modular helper function that can be overridden in case
        subclasses would like to modify training behavior for the
        action sampler.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

            epoch (int): epoch number - required by some Algos that need
                to perform staged training and early stopping

            no_backprop (bool): if True, don't perform any learning updates.

        Returns:
            info (dict): dictionary of relevant inputs, outputs, and losses
                that might be relevant for logging
            outputs (dict): dictionary of outputs to use during critic training
                (for computing target values)
        """
        info = OrderedDict()
        if self.algo_config.action_sampler.vae.prior.use_categorical:
            temperature = self.algo_config.action_sampler.vae.prior.categorical_init_temp - epoch * self.algo_config.action_sampler.vae.prior.categorical_temp_anneal_step
            temperature = max(temperature, self.algo_config.action_sampler.vae.prior.categorical_min_temp)
            self.nets["action_sampler"].set_gumbel_temperature(temperature)

        vae_inputs = dict(
            actions=batch["actions"],
            obs_dict=batch["obs"],
            goal_dict=batch["goal_obs"],
        )

        # maybe freeze encoder weights
        if (self.algo_config.action_sampler.freeze_encoder_epoch != -1) and (epoch >= self.algo_config.action_sampler.freeze_encoder_epoch):
            vae_inputs["freeze_encoder"] = True

        # VAE forward
        vae_outputs = self.nets["action_sampler"].forward_train(**vae_inputs)
        recons_loss = vae_outputs["reconstruction_loss"]
        kl_loss = vae_outputs["kl_loss"]
        vae_loss = recons_loss + self.algo_config.action_sampler.vae.kl_weight * kl_loss
        info["action_sampler/loss"] = vae_loss
        info["action_sampler/recons_loss"] = recons_loss
        info["action_sampler/kl_loss"] = kl_loss
        if not self.algo_config.action_sampler.vae.prior.use_categorical:
            with torch.no_grad():
                encoder_variance = torch.exp(vae_outputs["encoder_params"]["logvar"]).mean()
            info["action_sampler/encoder_variance"] = encoder_variance
        outputs = TensorUtils.detach(vae_outputs)

        # VAE gradient step
        if not no_backprop:
            vae_grad_norms = TorchUtils.backprop_for_loss(
                net=self.nets["action_sampler"],
                optim=self.optimizers["action_sampler"],
                loss=vae_loss,
            )
            info["action_sampler/grad_norms"] = vae_grad_norms
        return info, outputs

    def _train_critic_on_batch(self, batch, action_sampler_outputs, epoch, no_backprop=False):
        """
        A modular helper function that can be overridden in case
        subclasses would like to modify training behavior for the
        critics.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

            action_sampler_outputs (dict): dictionary of outputs from the action sampler. Used
                to form target values for training the critic

            epoch (int): epoch number - required by some Algos that need
                to perform staged training and early stopping

            no_backprop (bool): if True, don't perform any learning updates.

        Returns:
            info (dict): dictionary of relevant inputs, outputs, and losses
                that might be relevant for logging
            critic_outputs (dict): dictionary of critic outputs - useful for 
                logging purposes
        """
        info = OrderedDict()

        # batch variables
        s_batch = batch["obs"]
        a_batch = batch["actions"]
        r_batch = batch["rewards"]
        ns_batch = batch["next_obs"]
        goal_s_batch = batch["goal_obs"]

        # 1 if not done, 0 otherwise
        done_mask_batch = 1. - batch["dones"]
        info["done_masks"] = done_mask_batch

        # Bellman backup for Q-targets
        q_targets = self._get_target_values(
            next_states=ns_batch, 
            goal_states=goal_s_batch, 
            rewards=r_batch, 
            dones=done_mask_batch,
            action_sampler_outputs=action_sampler_outputs,
        )
        info["critic/q_targets"] = q_targets

        # Train all critics using this set of targets for regression
        critic_outputs = []
        for critic_ind, critic in enumerate(self.nets["critic"]):
            critic_loss, critic_output = self._compute_critic_loss(
                critic=critic, 
                states=s_batch, 
                actions=a_batch, 
                goal_states=goal_s_batch, 
                q_targets=q_targets,
            )
            info["critic/critic{}_loss".format(critic_ind + 1)] = critic_loss
            critic_outputs.append(critic_output)

            if not no_backprop:
                critic_grad_norms = TorchUtils.backprop_for_loss(
                    net=self.nets["critic"][critic_ind],
                    optim=self.optimizers["critic"][critic_ind],
                    loss=critic_loss, 
                    max_grad_norm=self.algo_config.critic.max_gradient_norm,
                )
                info["critic/critic{}_grad_norms".format(critic_ind + 1)] = critic_grad_norms

        return info, critic_outputs

    def _train_actor_on_batch(self, batch, action_sampler_outputs, critic_outputs, epoch, no_backprop=False):
        """
        A modular helper function that can be overridden in case
        subclasses would like to modify training behavior for the
        perturbation actor.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

            action_sampler_outputs (dict): dictionary of outputs from the action sampler. Currently
                unused, although more sophisticated models may use it.

            critic_outputs (dict): dictionary of outputs from the critic. Currently
                unused, although more sophisticated models may use it.

            epoch (int): epoch number - required by some Algos that need
                to perform staged training and early stopping

            no_backprop (bool): if True, don't perform any learning updates.

        Returns:
            info (dict): dictionary of relevant inputs, outputs, and losses
                that might be relevant for logging
        """
        assert self.algo_config.actor.enabled

        info = OrderedDict()

        # Actor loss (update with DDPG loss)
        s_batch = batch["obs"]
        goal_s_batch = batch["goal_obs"]

        # sample some actions from action sampler and perturb them, then improve perturbations
        # where improvement is measured by the critic
        sampled_actions = self.nets["action_sampler"](s_batch, goal_s_batch).detach() # don't backprop into samples
        perturbed_actions = self.nets["actor"](s_batch, sampled_actions, goal_s_batch)
        actor_loss = -(self.nets["critic"][0](s_batch, perturbed_actions, goal_s_batch)).mean()
        info["actor/loss"] = actor_loss

        if not no_backprop:
            actor_grad_norms = TorchUtils.backprop_for_loss(
                net=self.nets["actor"],
                optim=self.optimizers["actor"],
                loss=actor_loss,
            )
            info["actor/grad_norms"] = actor_grad_norms

        return info

    def _get_target_values(self, next_states, goal_states, rewards, dones, action_sampler_outputs=None):
        """
        Helper function to get target values for training Q-function with TD-loss.

        Args:
            next_states (dict): batch of next observations
            goal_states (dict): if not None, batch of goal observations
            rewards (torch.Tensor): batch of rewards - should be shape (B, 1)
            dones (torch.Tensor): batch of done signals - should be shape (B, 1)
            action_sampler_outputs (dict): dictionary of outputs from the action sampler. Currently
                unused, although more sophisticated models may use it.

        Returns:
            q_targets (torch.Tensor): target Q-values to use for TD loss
        """

        with torch.no_grad():
            # we need to stack the observations with redundancy @num_action_samples here, then decode 
            # to get all sampled actions. for example, if we generate 2 samples per observation and
            # the batch size is 3, then ob_tiled = [ob1; ob1; ob2; ob2; ob3; ob3]
            next_states_tiled = ObsUtils.repeat_and_stack_observation(next_states, n=self.algo_config.critic.num_action_samples)
            goal_states_tiled = None
            if len(self.goal_shapes) > 0:
                goal_states_tiled = ObsUtils.repeat_and_stack_observation(goal_states, n=self.algo_config.critic.num_action_samples)

            # sample action proposals
            next_sampled_actions = self._sample_actions_for_value_maximization(
                states_tiled=next_states_tiled,
                goal_states_tiled=goal_states_tiled,
                for_target_update=True,
            )

            q_targets = self._get_target_values_from_sampled_actions(
                next_states_tiled=next_states_tiled, 
                next_sampled_actions=next_sampled_actions, 
                goal_states_tiled=goal_states_tiled, 
                rewards=rewards, 
                dones=dones,
            )

        return q_targets

    def _sample_actions_for_value_maximization(self, states_tiled, goal_states_tiled, for_target_update):
        """
        Helper function to sample actions for maximization (the "batch-constrained" part of 
        batch-constrained q-learning).

        Args:
            states_tiled (dict): observations to use for sampling actions. Assumes that tiling
                has already occurred - so that if the batch size is B, and N samples are
                desired for each observation in the batch, the leading dimension for each
                observation in the dict is B * N

            goal_states_tiled (dict): if not None, goal observations

            for_target_update (bool): if True, actions are being sampled for use in training the
                critic - which means the target actor network should be used

        Returns:
            sampled_actions (torch.Tensor): actions sampled from the action sampler, and maybe
                perturbed by the actor network
        """

        with torch.no_grad():
            sampled_actions = self.nets["action_sampler"](states_tiled, goal_states_tiled)
            if self.algo_config.actor.enabled:
                actor = self.nets["actor"]
                if for_target_update:
                    actor = self.nets["actor_target"]
                # perturb the actions with the policy
                sampled_actions = actor(states_tiled, sampled_actions, goal_states_tiled)

        return sampled_actions

    def _get_target_values_from_sampled_actions(self, next_states_tiled, next_sampled_actions, goal_states_tiled, rewards, dones):
        """
        Helper function to get target values for training Q-function with TD-loss. The function
        assumes that action candidates to maximize over have already been computed, and that
        the input states have been tiled (repeated) to be compatible with the sampled actions.

        Args:
            next_states_tiled (dict): next observations to use for sampling actions. Assumes that 
                tiling has already occurred - so that if the batch size is B, and N samples are
                desired for each observation in the batch, the leading dimension for each
                observation in the dict is B * N

            next_sampled_actions (torch.Tensor): actions sampled from the action sampler. This function
                will maximize the critic over these action candidates (using the TD3 trick)

            goal_states_tiled (dict): if not None, goal observations

            rewards (torch.Tensor): batch of rewards - should be shape (B, 1)

            dones (torch.Tensor): batch of done signals - should be shape (B, 1)

        Returns:
            q_targets (torch.Tensor): target Q-values to use for TD loss
        """
        with torch.no_grad():
            # feed tiled observations and sampled actions into the critics and then
            # reshape to get all Q-values in second dimension per observation in batch.
            all_value_targets = self.nets["critic_target"][0](next_states_tiled, next_sampled_actions, goal_states_tiled).reshape(
                -1, self.algo_config.critic.num_action_samples)
            max_value_targets = all_value_targets
            min_value_targets = all_value_targets

            # TD3 trick to combine max and min over all Q-ensemble estimates into single target estimates
            for critic_target in self.nets["critic_target"][1:]:
                all_value_targets = critic_target(next_states_tiled, next_sampled_actions, goal_states_tiled).reshape(
                    -1, self.algo_config.critic.num_action_samples)
                max_value_targets = torch.max(max_value_targets, all_value_targets)
                min_value_targets = torch.min(min_value_targets, all_value_targets)
            all_value_targets = self.algo_config.critic.ensemble.weight * min_value_targets + \
                                (1. - self.algo_config.critic.ensemble.weight) * max_value_targets

            # take maximum over all sampled action values per observation and compute targets
            value_targets = torch.max(all_value_targets, dim=1, keepdim=True)[0]
            q_targets = rewards + dones * self.discount * value_targets

        return q_targets

    def _compute_critic_loss(self, critic, states, actions, goal_states, q_targets):
        """
        Helper function to compute loss between estimated Q-values and target Q-values.
        It should also return outputs needed for downstream training (for training the
        actor).

        Args:
            critic (torch.nn.Module): critic network
            states (dict): batch of observations
            actions (torch.Tensor): batch of actions
            goal_states (dict): if not None, batch of goal observations
            q_targets (torch.Tensor): batch of target q-values for the TD loss

        Returns:
            critic_loss (torch.Tensor): critic loss
            critic_output (dict): additional outputs from the critic. This function
                returns None, but subclasses may want to provide some information
                here.
        """
        q_estimated = critic(states, actions, goal_states)
        if self.algo_config.critic.use_huber:
            critic_loss = nn.SmoothL1Loss()(q_estimated, q_targets)
        else:
            critic_loss = nn.MSELoss()(q_estimated, q_targets)
        return critic_loss, None

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
            info = PolicyAlgo.train_on_batch(self, batch, epoch, validate=validate)

            # Action Sampler training
            no_action_sampler_backprop = validate or (not self._check_epoch(net_name="action_sampler", epoch=epoch))
            with TorchUtils.maybe_no_grad(no_grad=no_action_sampler_backprop):
                action_sampler_info, action_sampler_outputs = self._train_action_sampler_on_batch(
                    batch=batch, 
                    epoch=epoch, 
                    no_backprop=no_action_sampler_backprop,
                )
            info.update(action_sampler_info)

            # make sure action sampler is in eval mode for models like GMM which may require low-noise
            # samples when sampling actions.
            self.nets["action_sampler"].eval()

            # Critic training
            no_critic_backprop = validate or (not self._check_epoch(net_name="critic", epoch=epoch))
            with TorchUtils.maybe_no_grad(no_grad=no_critic_backprop):
                critic_info, critic_outputs = self._train_critic_on_batch(
                    batch=batch, 
                    action_sampler_outputs=action_sampler_outputs,
                    epoch=epoch, 
                    no_backprop=no_critic_backprop,
                )
            info.update(critic_info)

            if self.algo_config.actor.enabled:
                # Actor training
                no_actor_backprop = validate or (not self._check_epoch(net_name="actor", epoch=epoch))
                with TorchUtils.maybe_no_grad(no_grad=no_actor_backprop):
                    actor_info = self._train_actor_on_batch(
                        batch=batch, 
                        action_sampler_outputs=action_sampler_outputs, 
                        critic_outputs=critic_outputs, 
                        epoch=epoch, 
                        no_backprop=no_actor_backprop,
                    )
                info.update(actor_info)

            if not validate:
                # restore to train mode if necessary
                self.nets["action_sampler"].train()

            # update the target critic networks (only when critic has gradient update)
            if not no_critic_backprop:
                with torch.no_grad():
                    for critic_ind in range(len(self.nets["critic"])):
                        TorchUtils.soft_update(
                            source=self.nets["critic"][critic_ind], 
                            target=self.nets["critic_target"][critic_ind], 
                            tau=self.algo_config.target_tau,
                        )

            # update target actor network (only when actor has gradient update)
            if self.algo_config.actor.enabled and (not no_actor_backprop):
                with torch.no_grad():
                    TorchUtils.soft_update(
                        source=self.nets["actor"], 
                        target=self.nets["actor_target"], 
                        tau=self.algo_config.target_tau,
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
        loss_log = OrderedDict()

        # record current optimizer learning rates
        for k in self.optimizers:
            keys = [k]
            optims = [self.optimizers[k]]
            if k == "critic":
                # account for critic having one optimizer per ensemble member
                keys = ["{}{}".format(k, critic_ind) for critic_ind in range(len(self.nets["critic"]))]
                optims = self.optimizers[k]
            for kp, optimizer in zip(keys, optims):
                for i, param_group in enumerate(optimizer.param_groups):
                    loss_log["Optimizer/{}{}_lr".format(kp, i)] = param_group["lr"]

        # extract relevant logs for action sampler, critic, and actor
        loss_log["Loss"] = 0.
        for loss_logger in [self._log_action_sampler_info, self._log_critic_info, self._log_actor_info]:
            this_log = loss_logger(info)
            if "Loss" in this_log:
                # manually merge total loss
                loss_log["Loss"] += this_log["Loss"]
                del this_log["Loss"]
            loss_log.update(this_log)

        return loss_log

    def _log_action_sampler_info(self, info):
        """
        Helper function to extract action sampler-relevant information for logging.
        """
        loss_log = OrderedDict()
        loss_log["Action_Sampler/Loss"] = info["action_sampler/loss"].item()
        loss_log["Action_Sampler/Reconsruction_Loss"] = info["action_sampler/recons_loss"].item()
        loss_log["Action_Sampler/KL_Loss"] = info["action_sampler/kl_loss"].item()
        if self.algo_config.action_sampler.vae.prior.use_categorical:
            loss_log["Action_Sampler/Gumbel_Temperature"] = self.nets["action_sampler"].get_gumbel_temperature()
        else:
            loss_log["Action_Sampler/Encoder_Variance"] = info["action_sampler/encoder_variance"].item()
        if "action_sampler/grad_norms" in info:
            loss_log["Action_Sampler/Grad_Norms"] = info["action_sampler/grad_norms"]
        loss_log["Loss"] = loss_log["Action_Sampler/Loss"]
        return loss_log

    def _log_critic_info(self, info):
        """
        Helper function to extract critic-relevant information for logging.
        """
        loss_log = OrderedDict()
        if "done_masks" in info:
            loss_log["Critic/Done_Mask_Percentage"] = 100. * torch.mean(info["done_masks"]).item()
        if "critic/q_targets" in info:
            loss_log["Critic/Q_Targets"] = info["critic/q_targets"].mean().item()
        loss_log["Loss"] = 0.
        for critic_ind in range(len(self.nets["critic"])):
            loss_log["Critic/Critic{}_Loss".format(critic_ind + 1)] = info["critic/critic{}_loss".format(critic_ind + 1)].item()
            if "critic/critic{}_grad_norms".format(critic_ind + 1) in info:
                loss_log["Critic/Critic{}_Grad_Norms".format(critic_ind + 1)] = info["critic/critic{}_grad_norms".format(critic_ind + 1)]
            loss_log["Loss"] += loss_log["Critic/Critic{}_Loss".format(critic_ind + 1)]
        return loss_log

    def _log_actor_info(self, info):
        """
        Helper function to extract actor-relevant information for logging.
        """
        loss_log = OrderedDict()
        if self.algo_config.actor.enabled:
            loss_log["Actor/Loss"] = info["actor/loss"].item()
            if "actor/grad_norms" in info:
                loss_log["Actor/Grad_Norms"] = info["actor/grad_norms"]
            loss_log["Loss"] = loss_log["Actor/Loss"]
        return loss_log

    def set_train(self):
        """
        Prepare networks for evaluation. Update from super class to make sure
        target networks stay in evaluation mode all the time.
        """
        self.nets.train()

        # target networks always in eval
        for critic_ind in range(len(self.nets["critic_target"])):
            self.nets["critic_target"][critic_ind].eval()

        if self.algo_config.actor.enabled:
            self.nets["actor_target"].eval()

    def on_epoch_end(self, epoch):
        """
        Called at the end of each epoch.
        """

        # LR scheduling updates
        for lr_sc in self.lr_schedulers["critic"]:
            if lr_sc is not None:
                lr_sc.step()

        if self.lr_schedulers["action_sampler"] is not None:
            self.lr_schedulers["action_sampler"].step()

        if self.algo_config.actor.enabled and self.lr_schedulers["actor"] is not None:
            self.lr_schedulers["actor"].step()

    def _get_best_value(self, obs_dict, goal_dict=None):
        """
        Internal helper function for getting the best value for a given state and 
        the corresponding best action. Meant to be used at test-time. Key differences 
        between this and retrieving target values at train-time are that (1) only a 
        single critic is used for the value estimate and (2) the critic and actor 
        are used instead of the target critic and target actor.

        Args:
            obs_dict (dict): batch of current observations
            goal_dict (dict): (optional) goal

        Returns:
            best_value (torch.Tensor): best values
            best_action (torch.Tensor): best actions
        """
        assert not self.nets.training

        random_key = list(obs_dict.keys())[0]
        batch_size = obs_dict[random_key].shape[0]

        # number of action proposals from action sampler
        num_action_samples = self.algo_config.critic.num_action_samples_rollout

        # we need to stack the observations with redundancy @num_action_samples here, then decode 
        # to get all sampled actions. for example, if we generate 2 samples per observation and
        # the batch size is 3, then ob_tiled = [ob1; ob1; ob2; ob2; ob3; ob3]
        ob_tiled = ObsUtils.repeat_and_stack_observation(obs_dict, n=num_action_samples)
        goal_tiled = None
        if len(self.goal_shapes) > 0:
            goal_tiled = ObsUtils.repeat_and_stack_observation(goal_dict, n=num_action_samples)

        sampled_actions = self._sample_actions_for_value_maximization(
            states_tiled=ob_tiled, 
            goal_states_tiled=goal_tiled,
            for_target_update=False,
        )

        # feed tiled observations and perturbed sampled actions into the critic and then
        # reshape to get all Q-values in second dimension per observation in batch.
        # finally, just take a maximum across that second dimension to take the best sampled action
        all_critic_values = self.nets["critic"][0](ob_tiled, sampled_actions, goal_tiled).reshape(-1, num_action_samples)
        best_action_index = torch.argmax(all_critic_values, dim=1)

        all_actions = sampled_actions.reshape(batch_size, num_action_samples, -1)
        best_action = all_actions[torch.arange(all_actions.shape[0]), best_action_index]
        best_value = all_critic_values[torch.arange(all_critic_values.shape[0]), best_action_index].unsqueeze(1)

        return best_value, best_action

    def get_action(self, obs_dict, goal_dict=None):
        """
        Get policy action outputs.

        Args:
            obs_dict (dict): current observation
            goal_dict (dict): (optional) goal

        Returns:
            action (torch.Tensor): action tensor
        """
        assert not self.nets.training

        _, best_action = self._get_best_value(obs_dict=obs_dict, goal_dict=goal_dict)
        return best_action

    def get_state_value(self, obs_dict, goal_dict=None):
        """
        Get state value outputs.

        Args:
            obs_dict (dict): current observation
            goal_dict (dict): (optional) goal

        Returns:
            value (torch.Tensor): value tensor
        """
        assert not self.nets.training

        best_value, _ = self._get_best_value(obs_dict=obs_dict, goal_dict=goal_dict)
        return best_value

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
        assert not self.nets.training

        return self.nets["critic"][0](obs_dict, actions, goal_dict)


class BCQ_GMM(BCQ):
    """
    A simple modification to BCQ that replaces the VAE used to sample action proposals from the
    batch with a GMM.
    """
    def _create_action_sampler(self):
        """
        Called in @_create_networks to make action sampler network.
        """
        assert self.algo_config.action_sampler.gmm.enabled

        # GMM network for approximate sampling from batch dataset
        self.nets["action_sampler"] = PolicyNets.GMMActorNetwork(
            obs_shapes=self.obs_shapes,
            goal_shapes=self.goal_shapes,
            ac_dim=self.ac_dim,
            mlp_layer_dims=self.algo_config.action_sampler.actor_layer_dims,
            num_modes=self.algo_config.action_sampler.gmm.num_modes,
            min_std=self.algo_config.action_sampler.gmm.min_std,
            std_activation=self.algo_config.action_sampler.gmm.std_activation,
            low_noise_eval=self.algo_config.action_sampler.gmm.low_noise_eval,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
        )

    def _train_action_sampler_on_batch(self, batch, epoch, no_backprop=False):
        """
        Modify this helper function from superclass to train GMM action sampler
        with maximum likelihood.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

            epoch (int): epoch number - required by some Algos that need
                to perform staged training and early stopping

            no_backprop (bool): if True, don't perform any learning updates.

        Returns:
            info (dict): dictionary of relevant inputs, outputs, and losses
                that might be relevant for logging
            outputs (dict): dictionary of outputs to use during critic training
                (for computing target values)
        """
        info = OrderedDict()

        # GMM forward
        dists = self.nets["action_sampler"].forward_train(
            obs_dict=batch["obs"], 
            goal_dict=batch["goal_obs"],
        )

        # make sure that this is a batch of multivariate action distributions, so that
        # the log probability computation will be correct
        assert len(dists.batch_shape) == 1
        log_probs = dists.log_prob(batch["actions"])
        loss = -log_probs.mean()
        info["action_sampler/loss"] = loss

        # GMM gradient step
        if not no_backprop:
            gmm_grad_norms = TorchUtils.backprop_for_loss(
                net=self.nets["action_sampler"],
                optim=self.optimizers["action_sampler"],
                loss=loss,
            )
            info["action_sampler/grad_norms"] = gmm_grad_norms
        return info, None

    def _log_action_sampler_info(self, info):
        """
        Update from superclass for GMM (no KL loss).
        """
        loss_log = OrderedDict()
        loss_log["Action_Sampler/Loss"] = info["action_sampler/loss"].item()
        if "action_sampler/grad_norms" in info:
            loss_log["Action_Sampler/Grad_Norms"] = info["action_sampler/grad_norms"]
        loss_log["Loss"] = loss_log["Action_Sampler/Loss"]
        return loss_log


class BCQ_Distributional(BCQ):
    """
    BCQ with distributional critics. Distributional critics output categorical
    distributions over a discrete set of values instead of expected returns.
    Some parts of this implementation were adapted from ACME (https://github.com/deepmind/acme).
    """
    def _create_critics(self):
        """
        Called in @_create_networks to make critic networks.
        """
        assert self.algo_config.critic.distributional.enabled
        critic_class = ValueNets.DistributionalActionValueNetwork
        critic_args = dict(
            obs_shapes=self.obs_shapes,
            ac_dim=self.ac_dim,
            mlp_layer_dims=self.algo_config.critic.layer_dims,
            value_bounds=self.algo_config.critic.value_bounds,
            num_atoms=self.algo_config.critic.distributional.num_atoms,
            goal_shapes=self.goal_shapes,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
        )

        # Q network ensemble and target ensemble
        self.nets["critic"] = nn.ModuleList()
        self.nets["critic_target"] = nn.ModuleList()

        # NOTE: ensemble value in config is ignored, and only 1 critic is used.
        critic = critic_class(**critic_args)
        self.nets["critic"].append(critic)

        critic_target = critic_class(**critic_args)
        self.nets["critic_target"].append(critic_target)

    def _get_target_values_from_sampled_actions(self, next_states_tiled, next_sampled_actions, goal_states_tiled, rewards, dones):
        """
        Helper function to get target values for training Q-function with TD-loss. Update from superclass
        to account for distributional value functions.

        Args:
            next_states_tiled (dict): next observations to use for sampling actions. Assumes that 
                tiling has already occurred - so that if the batch size is B, and N samples are
                desired for each observation in the batch, the leading dimension for each
                observation in the dict is B * N

            next_sampled_actions (torch.Tensor): actions sampled from the action sampler. This function
                will maximize the critic over these action candidates (using the TD3 trick)

            goal_states_tiled (dict): if not None, goal observations

            rewards (torch.Tensor): batch of rewards - should be shape (B, 1)

            dones (torch.Tensor): batch of done signals - should be shape (B, 1)

        Returns:
            target_categorical_probabilities (torch.Tensor): target categorical probabilities
                to use in the bellman backup
        """

        with torch.no_grad():
            # compute expected returns of the sampled actions and maximize to find the best action
            all_vds = self.nets["critic_target"][0].forward_train(next_states_tiled, next_sampled_actions, goal_states_tiled)
            expected_values = all_vds.mean().reshape(-1, self.algo_config.critic.num_action_samples)
            best_action_index = torch.argmax(expected_values, dim=1)
            all_actions = next_sampled_actions.reshape(-1, self.algo_config.critic.num_action_samples, self.ac_dim)
            best_action = all_actions[torch.arange(all_actions.shape[0]), best_action_index]

            # get the corresponding probabilities for the categorical distributions corresponding to the best actions
            all_vd_probs = all_vds.probs.reshape(-1, self.algo_config.critic.num_action_samples, self.algo_config.critic.distributional.num_atoms)
            target_vd_probs = all_vd_probs[torch.arange(all_vd_probs.shape[0]), best_action_index]

            # bellman backup to get a new grid of values - then project onto the canonical atoms to obtain a
            # target set of categorical probabilities over the atoms
            atom_value_grid = all_vds.values
            target_value_grid = rewards + dones * self.discount * atom_value_grid
            target_categorical_probabilities = LossUtils.project_values_onto_atoms(
                values=target_value_grid,
                probabilities=target_vd_probs,
                atoms=atom_value_grid,
            )

        return target_categorical_probabilities

    def _compute_critic_loss(self, critic, states, actions, goal_states, q_targets):
        """
        Overrides super class to compute a distributional loss. Since values are
        categorical distributions, this is just computing a cross-entropy
        loss between the two distributions.

        NOTE: q_targets is expected to be a batch of normalized probability vectors that correspond to
              the target categorical distributions over the value atoms.

        Args:
            critic (torch.nn.Module): critic network
            states (dict): batch of observations
            actions (torch.Tensor): batch of actions
            goal_states (dict): if not None, batch of goal observations
            q_targets (torch.Tensor): batch of target q-values for the TD loss

        Returns:
            critic_loss (torch.Tensor): critic loss
            critic_output (dict): additional outputs from the critic. This function
                returns None, but subclasses may want to provide some information
                here.
        """

        # this should be the equivalent of softmax with logits from tf
        vd = critic.forward_train(states, actions, goal_states)
        log_probs = F.log_softmax(vd.logits, dim=-1)
        critic_loss = nn.KLDivLoss(reduction='batchmean')(log_probs, q_targets)
        return critic_loss, None
