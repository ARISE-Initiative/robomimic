"""
Implementation of TD3-BC. 
Based on https://github.com/sfujim/TD3_BC
(Paper - https://arxiv.org/abs/1812.02900).

Note that several parts are exactly the same as the BCQ implementation,
such as @_create_critics, @process_batch_for_training, and 
@_train_critic_on_batch. They are replicated here (instead of subclassing 
from the BCQ algo class) to be explicit and have implementation details 
self-contained in this file.
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


@register_algo_factory_func("td3_bc")
def algo_config_to_class(algo_config):
    """
    Maps algo config to the TD3_BC algo class to instantiate, along with additional algo kwargs.

    Args:
        algo_config (Config instance): algo config

    Returns:
        algo_class: subclass of Algo
        algo_kwargs (dict): dictionary of additional kwargs to pass to algorithm
    """
    # only one variant of TD3_BC for now
    return TD3_BC, {}


class TD3_BC(PolicyAlgo, ValueAlgo):
    """
    Default TD3_BC training, based on https://arxiv.org/abs/2106.06860 and
    https://github.com/sfujim/TD3_BC.
    """
    def __init__(self, **kwargs):
        PolicyAlgo.__init__(self, **kwargs)

        # save the discount factor - it may be overriden later
        self.set_discount(self.algo_config.discount)

        # initialize actor update counter. This is used to train the actor at a lower freq than critic
        self.actor_update_counter = 0

    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        self.nets = nn.ModuleDict()

        self._create_critics()
        self._create_actor()

        # sync target networks at beginning of training
        with torch.no_grad():
            for critic_ind in range(len(self.nets["critic"])):
                TorchUtils.hard_update(
                    source=self.nets["critic"][critic_ind], 
                    target=self.nets["critic_target"][critic_ind],
                )

            TorchUtils.hard_update(
                source=self.nets["actor"], 
                target=self.nets["actor_target"],
            )

        self.nets = self.nets.float().to(self.device)

    def _create_critics(self):
        """
        Called in @_create_networks to make critic networks.

        Exactly the same as BCQ.
        """
        critic_class = ValueNets.ActionValueNetwork
        critic_args = dict(
            obs_shapes=self.obs_shapes,
            ac_dim=self.ac_dim,
            mlp_layer_dims=self.algo_config.critic.layer_dims,
            value_bounds=self.algo_config.critic.value_bounds,
            goal_shapes=self.goal_shapes,
            **ObsNets.obs_encoder_args_from_config(self.obs_config.encoder),
        )

        # Q network ensemble and target ensemble
        self.nets["critic"] = nn.ModuleList()
        self.nets["critic_target"] = nn.ModuleList()
        for _ in range(self.algo_config.critic.ensemble.n):
            critic = critic_class(**critic_args)
            self.nets["critic"].append(critic)

            critic_target = critic_class(**critic_args)
            self.nets["critic_target"].append(critic_target)

    def _create_actor(self):
        """
        Called in @_create_networks to make actor network.
        """
        actor_class = PolicyNets.ActorNetwork
        actor_args = dict(
            obs_shapes=self.obs_shapes,
            goal_shapes=self.goal_shapes,
            ac_dim=self.ac_dim,
            mlp_layer_dims=self.algo_config.actor.layer_dims,
            **ObsNets.obs_encoder_args_from_config(self.obs_config.encoder),
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

        Exactly the same as BCQ.

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

        return TensorUtils.to_device(TensorUtils.to_float(input_batch), self.device)

    def _train_critic_on_batch(self, batch, epoch, no_backprop=False):
        """
        A modular helper function that can be overridden in case
        subclasses would like to modify training behavior for the
        critics.

        Exactly the same as BCQ (except for removal of @action_sampler_outputs and @critic_outputs)

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

            epoch (int): epoch number - required by some Algos that need
                to perform staged training and early stopping

            no_backprop (bool): if True, don't perform any learning updates.

        Returns:
            info (dict): dictionary of relevant inputs, outputs, and losses
                that might be relevant for logging
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
        )
        info["critic/q_targets"] = q_targets

        # Train all critics using this set of targets for regression
        for critic_ind, critic in enumerate(self.nets["critic"]):
            critic_loss = self._compute_critic_loss(
                critic=critic, 
                states=s_batch, 
                actions=a_batch, 
                goal_states=goal_s_batch, 
                q_targets=q_targets,
            )
            info["critic/critic{}_loss".format(critic_ind + 1)] = critic_loss

            if not no_backprop:
                critic_grad_norms = TorchUtils.backprop_for_loss(
                    net=self.nets["critic"][critic_ind],
                    optim=self.optimizers["critic"][critic_ind],
                    loss=critic_loss, 
                    max_grad_norm=self.algo_config.critic.max_gradient_norm,
                )
                info["critic/critic{}_grad_norms".format(critic_ind + 1)] = critic_grad_norms

        return info

    def _train_actor_on_batch(self, batch, epoch, no_backprop=False):
        """
        A modular helper function that can be overridden in case
        subclasses would like to modify training behavior for the
        actor.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

            epoch (int): epoch number - required by some Algos that need
                to perform staged training and early stopping

            no_backprop (bool): if True, don't perform any learning updates.

        Returns:
            info (dict): dictionary of relevant inputs, outputs, and losses
                that might be relevant for logging
        """
        info = OrderedDict()

        # Actor loss (update with mixture of DDPG loss and BC loss)
        s_batch = batch["obs"]
        a_batch = batch["actions"]
        goal_s_batch = batch["goal_obs"]

        # lambda mixture weight is combination of hyperparameter (alpha) and Q-value normalization
        actor_actions = self.nets["actor"](s_batch, goal_s_batch)
        Q_values = self.nets["critic"][0](s_batch, actor_actions, goal_s_batch)
        lam = self.algo_config.alpha / Q_values.abs().mean().detach()
        actor_loss = -lam * Q_values.mean() + nn.MSELoss()(actor_actions, a_batch)
        info["actor/loss"] = actor_loss

        if not no_backprop:
            actor_grad_norms = TorchUtils.backprop_for_loss(
                net=self.nets["actor"],
                optim=self.optimizers["actor"],
                loss=actor_loss,
            )
            info["actor/grad_norms"] = actor_grad_norms

        return info

    def _get_target_values(self, next_states, goal_states, rewards, dones):
        """
        Helper function to get target values for training Q-function with TD-loss.

        Args:
            next_states (dict): batch of next observations
            goal_states (dict): if not None, batch of goal observations
            rewards (torch.Tensor): batch of rewards - should be shape (B, 1)
            dones (torch.Tensor): batch of done signals - should be shape (B, 1)

        Returns:
            q_targets (torch.Tensor): target Q-values to use for TD loss
        """

        with torch.no_grad():
            # get next actions via target actor and noise
            next_target_actions = self.nets["actor_target"](next_states, goal_states)
            noise = (
                torch.randn_like(next_target_actions) * self.algo_config.actor.noise_std
            ).clamp(-self.algo_config.actor.noise_clip, self.algo_config.actor.noise_clip)
            next_actions = (next_target_actions + noise).clamp(-1.0, 1.0)

            # TD3 trick to combine max and min over all Q-ensemble estimates into single target estimates
            all_value_targets = self.nets["critic_target"][0](next_states, next_actions, goal_states).reshape(-1, 1)
            max_value_targets = all_value_targets
            min_value_targets = all_value_targets
            for critic_target in self.nets["critic_target"][1:]:
                all_value_targets = critic_target(next_states, next_actions, goal_states).reshape(-1, 1)
                max_value_targets = torch.max(max_value_targets, all_value_targets)
                min_value_targets = torch.min(min_value_targets, all_value_targets)
            value_targets = self.algo_config.critic.ensemble.weight * min_value_targets + \
                                (1. - self.algo_config.critic.ensemble.weight) * max_value_targets
            q_targets = rewards + dones * self.discount * value_targets

        return q_targets

    def _compute_critic_loss(self, critic, states, actions, goal_states, q_targets):
        """
        Helper function to compute loss between estimated Q-values and target Q-values.

        Nearly the same as BCQ (return type slightly different).

        Args:
            critic (torch.nn.Module): critic network
            states (dict): batch of observations
            actions (torch.Tensor): batch of actions
            goal_states (dict): if not None, batch of goal observations
            q_targets (torch.Tensor): batch of target q-values for the TD loss

        Returns:
            critic_loss (torch.Tensor): critic loss
        """
        q_estimated = critic(states, actions, goal_states)
        if self.algo_config.critic.use_huber:
            critic_loss = nn.SmoothL1Loss()(q_estimated, q_targets)
        else:
            critic_loss = nn.MSELoss()(q_estimated, q_targets)
        return critic_loss

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

            # Critic training
            no_critic_backprop = validate or (not self._check_epoch(net_name="critic", epoch=epoch))
            with TorchUtils.maybe_no_grad(no_grad=no_critic_backprop):
                critic_info = self._train_critic_on_batch(
                    batch=batch, 
                    epoch=epoch, 
                    no_backprop=no_critic_backprop,
                )
            info.update(critic_info)

            # update actor and target networks at lower frequency
            if not no_critic_backprop:
                # update counter only on critic training gradient steps
                self.actor_update_counter += 1
            do_actor_update = (self.actor_update_counter % self.algo_config.actor.update_freq == 0)

            # Actor training
            no_actor_backprop = validate or (not self._check_epoch(net_name="actor", epoch=epoch))
            no_actor_backprop = no_actor_backprop or (not do_actor_update)
            with TorchUtils.maybe_no_grad(no_grad=no_actor_backprop):
                actor_info = self._train_actor_on_batch(
                    batch=batch, 
                    epoch=epoch, 
                    no_backprop=no_actor_backprop,
                )
            info.update(actor_info)

            if not no_actor_backprop:
                # to match original implementation, only update target networks on 
                # actor gradient steps
                with torch.no_grad():
                    # update the target critic networks
                    for critic_ind in range(len(self.nets["critic"])):
                        TorchUtils.soft_update(
                            source=self.nets["critic"][critic_ind], 
                            target=self.nets["critic_target"][critic_ind], 
                            tau=self.algo_config.target_tau,
                        )

                    # update target actor network
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

        # extract relevant logs for critic, and actor
        loss_log["Loss"] = 0.
        for loss_logger in [self._log_critic_info, self._log_actor_info]:
            this_log = loss_logger(info)
            if "Loss" in this_log:
                # manually merge total loss
                loss_log["Loss"] += this_log["Loss"]
                del this_log["Loss"]
            loss_log.update(this_log)

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

        self.nets["actor_target"].eval()

    def on_epoch_end(self, epoch):
        """
        Called at the end of each epoch.
        """

        # LR scheduling updates
        for lr_sc in self.lr_schedulers["critic"]:
            if lr_sc is not None:
                lr_sc.step()

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

        actions = self.nets["actor"](obs_dict=obs_dict, goal_dict=goal_dict)
        return self.nets["critic"][0](obs_dict, actions, goal_dict)

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
