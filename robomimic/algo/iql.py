"""
Implementation of Implicit Q-Learning (IQL).
Based off of https://github.com/rail-berkeley/rlkit/blob/master/rlkit/torch/sac/iql_trainer.py.
(Paper - https://arxiv.org/abs/2110.06169).
"""
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

import robomimic.models.policy_nets as PolicyNets
import robomimic.models.value_nets as ValueNets
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils
from robomimic.algo import register_algo_factory_func, ValueAlgo, PolicyAlgo


@register_algo_factory_func("iql")
def algo_config_to_class(algo_config):
    """
    Maps algo config to the IQL algo class to instantiate, along with additional algo kwargs.

    Args:
        algo_config (Config instance): algo config

    Returns:
        algo_class: subclass of Algo
        algo_kwargs (dict): dictionary of additional kwargs to pass to algorithm
    """
    return IQL, {}


class IQL(PolicyAlgo, ValueAlgo):
    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.

        Networks for this algo: critic (potentially ensemble), actor, value function
        """

        # Create nets
        self.nets = nn.ModuleDict()

        # Assemble args to pass to actor
        actor_args = dict(self.algo_config.actor.net.common)

        # Add network-specific args and define network class
        if self.algo_config.actor.net.type == "gaussian":
            actor_cls = PolicyNets.GaussianActorNetwork
            actor_args.update(dict(self.algo_config.actor.net.gaussian))
        elif self.algo_config.actor.net.type == "gmm":
            actor_cls = PolicyNets.GMMActorNetwork
            actor_args.update(dict(self.algo_config.actor.net.gmm))
        else:
            # Unsupported actor type!
            raise ValueError(f"Unsupported actor requested. "
                             f"Requested: {self.algo_config.actor.net.type}, "
                             f"valid options are: {['gaussian', 'gmm']}")

        # Actor
        self.nets["actor"] = actor_cls(
            obs_shapes=self.obs_shapes,
            goal_shapes=self.goal_shapes,
            ac_dim=self.ac_dim,
            mlp_layer_dims=self.algo_config.actor.layer_dims,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
            **actor_args,
        )

        # Critics
        self.nets["critic"] = nn.ModuleList()
        self.nets["critic_target"] = nn.ModuleList()
        for _ in range(self.algo_config.critic.ensemble.n):
            for net_list in (self.nets["critic"], self.nets["critic_target"]):
                critic = ValueNets.ActionValueNetwork(
                    obs_shapes=self.obs_shapes,
                    ac_dim=self.ac_dim,
                    mlp_layer_dims=self.algo_config.critic.layer_dims,
                    goal_shapes=self.goal_shapes,
                    encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
                )
                net_list.append(critic)

        # Value function network
        self.nets["vf"] = ValueNets.ValueNetwork(
            obs_shapes=self.obs_shapes,
            mlp_layer_dims=self.algo_config.critic.layer_dims,
            goal_shapes=self.goal_shapes,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
        )

        # Send networks to appropriate device
        self.nets = self.nets.float().to(self.device)

        # sync target networks at beginning of training
        with torch.no_grad():
            for critic, critic_target in zip(self.nets["critic"], self.nets["critic_target"]):
                TorchUtils.hard_update(
                    source=critic,
                    target=critic_target,
                )

    def process_batch_for_training(self, batch):
        """
        Processes input batch from a data loader to filter out relevant info and prepare the batch for training.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader

        Returns:
            input_batch (dict): processed and filtered batch that
                will be used for training
        """

        input_batch = dict()

        # remove temporal batches for all
        input_batch["obs"] = {k: batch["obs"][k][:, 0, :] for k in batch["obs"]}
        input_batch["next_obs"] = {k: batch["next_obs"][k][:, 0, :] for k in batch["next_obs"]}
        input_batch["goal_obs"] = batch.get("goal_obs", None) # goals may not be present
        input_batch["actions"] = batch["actions"][:, 0, :]
        input_batch["dones"] = batch["dones"][:, 0]
        input_batch["rewards"] = batch["rewards"][:, 0]

        return TensorUtils.to_device(TensorUtils.to_float(input_batch), self.device)

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
        info = OrderedDict()

        # Set the correct context for this training step
        with TorchUtils.maybe_no_grad(no_grad=validate):
            # Always run super call first
            info = super().train_on_batch(batch, epoch, validate=validate)

            # Compute loss for critic(s)
            critic_losses, vf_loss, critic_info = self._compute_critic_loss(batch)
            # Compute loss for actor
            actor_loss, actor_info = self._compute_actor_loss(batch, critic_info)

            if not validate:
                # Critic update
                self._update_critic(critic_losses, vf_loss)
                
                # Actor update
                self._update_actor(actor_loss)

            # Update info
            info.update(actor_info)
            info.update(critic_info)

        # Return stats
        return info

    def _compute_critic_loss(self, batch):
        """
        Helper function for computing Q and V losses. Called by @train_on_batch

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            critic_losses (list): list of critic (Q function) losses
            vf_loss (torch.Tensor): value function loss
            info (dict): dictionary of Q / V predictions and losses
        """
        info = OrderedDict()

        # get batch values
        obs = batch["obs"]
        actions = batch["actions"]
        next_obs = batch["next_obs"]
        goal_obs = batch["goal_obs"]
        rewards = torch.unsqueeze(batch["rewards"], 1)
        dones = torch.unsqueeze(batch["dones"], 1)

        # Q predictions
        pred_qs = [critic(obs_dict=obs, acts=actions, goal_dict=goal_obs)
                   for critic in self.nets["critic"]]

        info["critic/critic1_pred"] = pred_qs[0].mean()

        # Q target values
        target_vf_pred = self.nets["vf"](obs_dict=next_obs, goal_dict=goal_obs).detach()
        q_target = rewards + (1. - dones) * self.algo_config.discount * target_vf_pred
        q_target = q_target.detach()

        # Q losses
        critic_losses = []
        td_loss_fcn = nn.SmoothL1Loss() if self.algo_config.critic.use_huber else nn.MSELoss()
        for (i, q_pred) in enumerate(pred_qs):
            # Calculate td error loss
            td_loss = td_loss_fcn(q_pred, q_target)
            info[f"critic/critic{i+1}_loss"] = td_loss
            critic_losses.append(td_loss)

        # V predictions
        pred_qs = [critic(obs_dict=obs, acts=actions, goal_dict=goal_obs)
                        for critic in self.nets["critic_target"]]
        q_pred, _ = torch.cat(pred_qs, dim=1).min(dim=1, keepdim=True)
        q_pred = q_pred.detach()
        vf_pred = self.nets["vf"](obs_dict=obs, goal_dict=goal_obs)
        
        # V losses: expectile regression. see section 4.1 in https://arxiv.org/pdf/2110.06169.pdf
        vf_err = vf_pred - q_pred
        vf_sign = (vf_err > 0).float()
        vf_weight = (1 - vf_sign) * self.algo_config.vf_quantile + vf_sign * (1 - self.algo_config.vf_quantile)
        vf_loss = (vf_weight * (vf_err ** 2)).mean()
        
        # update logs for V loss
        info["vf/q_pred"] = q_pred
        info["vf/v_pred"] = vf_pred
        info["vf/v_loss"] = vf_loss

        # Return stats
        return critic_losses, vf_loss, info

    def _update_critic(self, critic_losses, vf_loss):
        """
        Helper function for updating critic and vf networks. Called by @train_on_batch

        Args:
            critic_losses (list): list of critic (Q function) losses
            vf_loss (torch.Tensor): value function loss
        """

        # update ensemble of critics
        for (critic_loss, critic, critic_target, optimizer) in zip(
                critic_losses, self.nets["critic"], self.nets["critic_target"], self.optimizers["critic"]
        ):
            TorchUtils.backprop_for_loss(
                net=critic,
                optim=optimizer,
                loss=critic_loss,
                max_grad_norm=self.algo_config.critic.max_gradient_norm,
                retain_graph=False,
            )

            # update target network
            with torch.no_grad():
                TorchUtils.soft_update(source=critic, target=critic_target, tau=self.algo_config.target_tau)

        # update V function network
        TorchUtils.backprop_for_loss(
            net=self.nets["vf"],
            optim=self.optimizers["vf"],
            loss=vf_loss,
            max_grad_norm=self.algo_config.critic.max_gradient_norm,
            retain_graph=False,
        )

    def _compute_actor_loss(self, batch, critic_info):
        """
        Helper function for computing actor loss. Called by @train_on_batch

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

            critic_info (dict): dictionary containing Q and V function predictions,
                to be used for computing advantage estimates

        Returns:
            actor_loss (torch.Tensor): actor loss
            info (dict): dictionary of actor losses, log_probs, advantages, and weights
        """
        info = OrderedDict()

        # compute log probability of batch actions
        dist = self.nets["actor"].forward_train(obs_dict=batch["obs"], goal_dict=batch["goal_obs"])
        log_prob = dist.log_prob(batch["actions"])

        info["actor/log_prob"] = log_prob.mean()

        # compute advantage estimate
        q_pred = critic_info["vf/q_pred"]
        v_pred = critic_info["vf/v_pred"]
        adv = q_pred - v_pred
        
        # compute weights
        weights = self._get_adv_weights(adv)

        # compute advantage weighted actor loss. disable gradients through weights
        actor_loss = (-log_prob * weights.detach()).mean()

        info["actor/loss"] = actor_loss

        # log adv-related values
        info["adv/adv"] = adv
        info["adv/adv_weight"] = weights

        # Return stats
        return actor_loss, info

    def _update_actor(self, actor_loss):
        """
        Helper function for updating actor network. Called by @train_on_batch

        Args:
            actor_loss (torch.Tensor): actor loss
        """

        TorchUtils.backprop_for_loss(
            net=self.nets["actor"],
            optim=self.optimizers["actor"],
            loss=actor_loss,
            max_grad_norm=self.algo_config.actor.max_gradient_norm,
        )
    
    def _get_adv_weights(self, adv):
        """
        Helper function for computing advantage weights. Called by @_compute_actor_loss

        Args:
            adv (torch.Tensor): raw advantage estimates

        Returns:
            weights (torch.Tensor): weights computed based on advantage estimates,
                in shape (B,) where B is batch size
        """
        
        # clip raw advantage values
        if self.algo_config.adv.clip_adv_value is not None:
            adv = adv.clamp(max=self.algo_config.adv.clip_adv_value)

        # compute weights based on advantage values
        beta = self.algo_config.adv.beta # temprature factor        
        weights = torch.exp(adv / beta)

        # clip final weights
        if self.algo_config.adv.use_final_clip is True:
            weights = weights.clamp(-100.0, 100.0)

        # reshape from (B, 1) to (B,)
        return weights[:, 0]

    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.

        Args:
            info (dict): dictionary of info

        Returns:
            loss_log (dict): name -> summary statistic
        """
        log = OrderedDict()

        log["actor/log_prob"] = info["actor/log_prob"].item()
        log["actor/loss"] = info["actor/loss"].item()

        log["critic/critic1_pred"] = info["critic/critic1_pred"].item()
        log["critic/critic1_loss"] = info["critic/critic1_loss"].item()

        log["vf/v_loss"] = info["vf/v_loss"].item()

        self._log_data_attributes(log, info, "vf/q_pred")
        self._log_data_attributes(log, info, "vf/v_pred")
        self._log_data_attributes(log, info, "adv/adv")
        self._log_data_attributes(log, info, "adv/adv_weight")

        return log

    def _log_data_attributes(self, log, info, key):
        """
        Helper function for logging statistics. Moodifies log in-place

        Args:
            log (dict): existing log dictionary
            log (dict): existing dictionary of tensors containing raw stats
            key (str): key to log
        """
        log[key + "/max"] = info[key].max().item()
        log[key + "/min"] = info[key].min().item()
        log[key + "/mean"] = info[key].mean().item()
        log[key + "/std"] = info[key].std().item()

    def on_epoch_end(self, epoch):
        """
        Called at the end of each epoch.
        """

        # LR scheduling updates
        for lr_sc in self.lr_schedulers["critic"]:
            if lr_sc is not None:
                lr_sc.step()

        if self.lr_schedulers["vf"] is not None:
            self.lr_schedulers["vf"].step()

        if self.lr_schedulers["actor"] is not None:
            self.lr_schedulers["actor"].step()

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

        return self.nets["actor"](obs_dict=obs_dict, goal_dict=goal_dict)