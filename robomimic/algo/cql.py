"""
Implementation of Conservative Q-Learning (CQL).
Based off of https://github.com/aviralkumar2907/CQL.
(Paper - https://arxiv.org/abs/2006.04779).
"""
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim

import robomimic.models.base_nets as BaseNets
import robomimic.models.obs_nets as ObsNets
import robomimic.models.policy_nets as PolicyNets
import robomimic.models.value_nets as ValueNets
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils
from robomimic.algo import register_algo_factory_func, ValueAlgo, PolicyAlgo


@register_algo_factory_func("cql")
def algo_config_to_class(algo_config):
    """
    Maps algo config to the CQL algo class to instantiate, along with additional algo kwargs.

    Args:
        algo_config (Config instance): algo config

    Returns:
        algo_class: subclass of Algo
        algo_kwargs (dict): dictionary of additional kwargs to pass to algorithm
    """
    return CQL, {}


class CQL(PolicyAlgo, ValueAlgo):
    """
    CQL-extension of SAC for the off-policy, offline setting. See https://arxiv.org/abs/2006.04779
    """
    def __init__(self, **kwargs):
        # Store entropy / cql settings first since the super init call requires them
        self.automatic_entropy_tuning = kwargs["algo_config"].actor.target_entropy is not None
        self.automatic_cql_tuning = kwargs["algo_config"].critic.target_q_gap is not None and \
                                    kwargs["algo_config"].critic.target_q_gap >= 0.0

        # Run super init first
        super().__init__(**kwargs)

        # Reward settings
        self.n_step = self.algo_config.n_step
        self.discount = self.algo_config.discount ** self.n_step

        # Now also store additional SAC- and CQL-specific stuff from the config
        self._num_batch_steps = 0
        self.bc_start_steps = self.algo_config.actor.bc_start_steps
        self.deterministic_backup = self.algo_config.critic.deterministic_backup
        self.td_loss_fcn = nn.SmoothL1Loss() if self.algo_config.critic.use_huber else nn.MSELoss()

        # Entropy settings
        self.target_entropy = -np.prod(self.ac_dim) if self.algo_config.actor.target_entropy in {None, "default"} else\
            self.algo_config.actor.target_entropy

        # CQL settings
        self.min_q_weight = self.algo_config.critic.min_q_weight
        self.target_q_gap = self.algo_config.critic.target_q_gap if self.automatic_cql_tuning else 0.0

    @property
    def log_entropy_weight(self):
        return self.nets["log_entropy_weight"]() if self.automatic_entropy_tuning else\
            torch.zeros(1, requires_grad=False, device=self.device)

    @property
    def log_cql_weight(self):
        return self.nets["log_cql_weight"]() if self.automatic_cql_tuning else\
            torch.log(torch.tensor(self.algo_config.critic.cql_weight, requires_grad=False, device=self.device))

    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.

        Networks for this algo: critic (potentially ensemble), policy
        """

        # Create nets
        self.nets = nn.ModuleDict()

        # Assemble args to pass to actor
        actor_args = dict(self.algo_config.actor.net.common)

        # Add network-specific args and define network class
        if self.algo_config.actor.net.type == "gaussian":
            actor_cls = PolicyNets.GaussianActorNetwork
            actor_args.update(dict(self.algo_config.actor.net.gaussian))
        else:
            # Unsupported actor type!
            raise ValueError(f"Unsupported actor requested. "
                             f"Requested: {self.algo_config.actor.net.type}, "
                             f"valid options are: {['gaussian']}")

        # Policy
        self.nets["actor"] = actor_cls(
            obs_shapes=self.obs_shapes,
            goal_shapes=self.goal_shapes,
            ac_dim=self.ac_dim,
            mlp_layer_dims=self.algo_config.actor.layer_dims,
            **actor_args,
            **ObsNets.obs_encoder_args_from_config(self.obs_config.encoder),
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
                    value_bounds=self.algo_config.critic.value_bounds,
                    goal_shapes=self.goal_shapes,
                    **ObsNets.obs_encoder_args_from_config(self.obs_config.encoder),
                )
                net_list.append(critic)

        # Entropy (if automatically tuning)
        if self.automatic_entropy_tuning:
            self.nets["log_entropy_weight"] = BaseNets.Parameter(torch.zeros(1))

        # CQL (if automatically tuning)
        if self.automatic_cql_tuning:
            self.nets["log_cql_weight"] = BaseNets.Parameter(torch.zeros(1))

        # Send networks to appropriate device
        self.nets = self.nets.float().to(self.device)

        # sync target networks at beginning of training
        with torch.no_grad():
            for critic, critic_target in zip(self.nets["critic"], self.nets["critic_target"]):
                TorchUtils.hard_update(
                    source=critic,
                    target=critic_target,
                )

    def _create_optimizers(self):
        """
        Creates optimizers using @self.optim_params and places them into @self.optimizers.

        Overrides base method since we might need to create aditional optimizers for the entropy
        and cql weight parameters (by default, the base class only creates optimizers for all
        entries in @self.nets that have corresponding entries in `self.optim_params` but these
        parameters do not).
        """

        # Create actor and critic optimizers via super method
        super()._create_optimizers()

        # We still need to potentially create additional optimizers based on algo settings

        # entropy (if automatically tuning)
        if self.automatic_entropy_tuning:
            self.optimizers["entropy"] = optim.Adam(
                params=self.nets["log_entropy_weight"].parameters(),
                lr=self.optim_params["actor"]["learning_rate"]["initial"],
                weight_decay=0.0,
            )

        # cql (if automatically tuning)
        if self.automatic_cql_tuning:
            self.optimizers["cql"] = optim.Adam(
                params=self.nets["log_cql_weight"].parameters(),
                lr=self.optim_params["critic"]["learning_rate"]["initial"],
                weight_decay=0.0,
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

        # Make sure the trajectory of actions received is greater than our step horizon
        assert batch["actions"].shape[1] >= self.n_step

        # remove temporal batches for all
        input_batch["obs"] = {k: batch["obs"][k][:, 0, :] for k in batch["obs"]}
        input_batch["next_obs"] = {k: batch["next_obs"][k][:, self.n_step - 1, :] for k in batch["next_obs"]}
        input_batch["goal_obs"] = batch.get("goal_obs", None) # goals may not be present
        input_batch["actions"] = batch["actions"][:, 0, :]

        # note: ensure scalar signals (rewards, done) retain last dimension of 1 to be compatible with model outputs

        # single timestep reward is discounted sum of intermediate rewards in sequence
        reward_seq = batch["rewards"][:, :self.n_step]
        discounts = torch.pow(self.algo_config.discount, torch.arange(self.n_step).float()).unsqueeze(0)
        input_batch["rewards"] = (reward_seq * discounts).sum(dim=1).unsqueeze(1)

        # consider this n-step seqeunce done if any intermediate dones are present
        done_seq = batch["dones"][:, :self.n_step]
        input_batch["dones"] = (done_seq.sum(dim=1) > 0).float().unsqueeze(1)

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
            super_info = super().train_on_batch(batch, epoch, validate=validate)
            # Train actor
            actor_info = self._train_policy_on_batch(batch, epoch, validate)
            # Train critic(s)
            critic_info = self._train_critic_on_batch(batch, epoch, validate)
            # Update info
            info.update(super_info)
            info.update(actor_info)
            info.update(critic_info)

        # Return stats
        return info

    def _train_policy_on_batch(self, batch, epoch, validate=False):
        """
        Training policy on a single batch of data.

        Loss is the ExpValue over sampled states of the (weighted) logprob of a sampled action
        under the current policy minus the Q value of associated with the (s, a) combo

        Intuitively, this tries to improve the odds of sampling actions with high Q values while simultaneously
        penalizing high probability actions.

        Since we're in the continuous setting, we monte carlo sample.

        Concretely:
            Loss = Average[ entropy_weight * logprob(f(eps; s) | s) - Q(s, f(eps; s) ]

            where we use the reparameterization trick with Gaussian function f(*) to parameterize
            actions as a function of the sampled noise param eps given input state s

        Additionally, we update the (log) entropy weight parameter if we're tuning that as well.

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

        # Sample actions from policy and get log probs
        dist = self.nets["actor"].forward_train(obs_dict=batch["obs"], goal_dict=batch["goal_obs"])
        actions, log_prob = self._get_actions_and_log_prob(dist=dist)

        # Calculate alpha
        entropy_weight_loss = -(self.log_entropy_weight * (log_prob + self.target_entropy).detach()).mean() if\
            self.automatic_entropy_tuning else 0.0
        entropy_weight = self.log_entropy_weight.exp()

        # Get predicted Q-values for all state, action pairs
        pred_qs = [critic(obs_dict=batch["obs"], acts=actions, goal_dict=batch["goal_obs"])
                   for critic in self.nets["critic"]]
        # We take the minimum for stability
        pred_qs, _ = torch.cat(pred_qs, dim=1).min(dim=1, keepdim=True)

        # Use BC if we're in the beginning of training, otherwise calculate policy loss normally
        baseline = dist.log_prob(batch["actions"]).unsqueeze(dim=-1) if\
            self._num_batch_steps < self.bc_start_steps else pred_qs
        policy_loss = (entropy_weight * log_prob - baseline).mean()

        # Add info
        info["entropy_weight"] = entropy_weight.item()
        info["entropy_weight_loss"] = entropy_weight_loss.item() if \
            self.automatic_entropy_tuning else entropy_weight_loss
        info["actor/loss"] = policy_loss

        # Take a training step if we're not validating
        if not validate:
            # Update batch step
            self._num_batch_steps += 1
            if self.automatic_entropy_tuning:
                # Alpha
                self.optimizers["entropy"].zero_grad()
                entropy_weight_loss.backward()
                self.optimizers["entropy"].step()
                info["entropy_grad_norms"] = self.log_entropy_weight.grad.data.norm(2).pow(2).item()

            # Policy
            actor_grad_norms = TorchUtils.backprop_for_loss(
                net=self.nets["actor"],
                optim=self.optimizers["actor"],
                loss=policy_loss,
                max_grad_norm=self.algo_config.actor.max_gradient_norm,
            )
            # Add info
            info["actor/grad_norms"] = actor_grad_norms

        # Return stats
        return info

    def _train_critic_on_batch(self, batch, epoch, validate=False):
        """
        Training critic(s) on a single batch of data.

        For a given batch of (s, a, r, s') tuples and n sampled actions (a_, a'_ corresponding to actions
        sampled from the learned policy at states s and s', respectively; a~ corresponding to uniformly random
        sampled actions):

            Loss = CQL_loss + SAC_loss

        Since we're in the continuous setting, we monte carlo sample for all ExpValues, which become Averages instead

        SAC_loss is the standard single-step TD error, corresponding to the following:

            SAC_loss = 0.5 * Average[ (Q(s,a) - (r + Average over a'_ [ Q(s', a'_) ]))^2 ]

        The CQL_loss corresponds to a weighted secondary objective, corresponding to the (ExpValue of Q values over
        sampled states and sampled actions from the LEARNED policy) minus the (ExpValue of Q values over
        sampled states and sampled actions from the DATASET policy) plus a regularizer as a function
        of the learned policy.

        Intuitively, this tries to penalize Q-values arbitrarily resulting from the learned policy (which may produce
        out-of-distribution (s,a) pairs) while preserving (known) Q-values taken from the dataset policy.

        As we are using SAC, we choose our regularizer to correspond to the negative KL divergence between our
        learned policy and a uniform distribution such that the first term in the CQL loss corresponds to the
        soft maximum over all Q values at any state s.

        For stability, we importance sample actions over random actions and from the current policy at s, s'.

        Moreover, if we want to tune the cql_weight automatically, we include the threshold value target_q_gap
        to penalize Q values that are overly-optimistic by the given threshold.

        In this case, the CQL_loss is as follows:

            CQL_loss = cql_weight * (Average [log (Average over a` in {a~, a_, a_'}: exp(Q(s,a`) - logprob(a`)) - Average [Q(s,a)]] - target_q_gap)

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
        B, A = batch["actions"].shape
        N = self.algo_config.critic.num_random_actions

        # Get predicted Q-values from taken actions
        q_preds = [critic(obs_dict=batch["obs"], acts=batch["actions"], goal_dict=batch["goal_obs"])
                   for critic in self.nets["critic"]]

        # Sample actions at the current and next step
        curr_dist = self.nets["actor"].forward_train(obs_dict=batch["obs"], goal_dict=batch["goal_obs"])
        next_dist = self.nets["actor"].forward_train(obs_dict=batch["next_obs"], goal_dict=batch["goal_obs"])
        next_actions, next_log_prob = self._get_actions_and_log_prob(dist=next_dist)

        # Don't capture gradients here, since the critic target network doesn't get trained (only soft updated)
        with torch.no_grad():
            # We take the max over all samples if the number of action samples is > 1
            if self.algo_config.critic.num_action_samples > 1:
                # Generate the target q values, using the backup from the next state
                temp_actions = next_dist.rsample(sample_shape=(self.algo_config.critic.num_action_samples,)).permute(1, 0, 2)
                target_qs = [self._get_qs_from_actions(
                    obs_dict=batch["next_obs"], actions=temp_actions, goal_dict=batch["goal_obs"], q_net=critic)
                                 .max(dim=1, keepdim=True)[0] for critic in self.nets["critic_target"]]
            else:
                target_qs = [critic(obs_dict=batch["next_obs"], acts=next_actions, goal_dict=batch["goal_obs"])
                             for critic in self.nets["critic_target"]]
            # Take the minimum over all critics
            target_qs, _ = torch.cat(target_qs, dim=1).min(dim=1, keepdim=True)
            # If only sampled once from each critic and not using a deterministic backup, subtract the logprob as well
            if self.algo_config.critic.num_action_samples == 1 and not self.deterministic_backup:
                target_qs = target_qs - self.log_entropy_weight.exp() * next_log_prob

            # Calculate the q target values
            done_mask_batch = 1. - batch["dones"]
            info["done_masks"] = done_mask_batch
            q_target = batch["rewards"] + done_mask_batch * self.discount * target_qs

        # Calculate CQL stuff
        cql_random_actions = torch.FloatTensor(N, B, A).uniform_(-1., 1.).to(self.device)                           # shape (N, B, A)
        cql_random_log_prob = np.log(0.5 ** A)
        cql_curr_actions, cql_curr_log_prob = self._get_actions_and_log_prob(dist=curr_dist, sample_shape=(N,))     # shape (N, B, A) and (N, B, 1)
        cql_next_actions, cql_next_log_prob = self._get_actions_and_log_prob(dist=next_dist, sample_shape=(N,))     # shape (N, B, A) and (N, B, 1)
        cql_curr_log_prob = cql_curr_log_prob.squeeze(dim=-1).permute(1, 0).detach()                                # shape (B, N)
        cql_next_log_prob = cql_next_log_prob.squeeze(dim=-1).permute(1, 0).detach()                                # shape (B, N)
        q_cats = []     # Each entry shape will be (B, N)

        for critic, q_pred in zip(self.nets["critic"], q_preds):
            # Compose Q values over all sampled actions (importance sampled)
            q_rand = self._get_qs_from_actions(obs_dict=batch["obs"], actions=cql_random_actions.permute(1, 0, 2), goal_dict=batch["goal_obs"], q_net=critic)
            q_curr = self._get_qs_from_actions(obs_dict=batch["obs"], actions=cql_curr_actions.permute(1, 0, 2), goal_dict=batch["goal_obs"], q_net=critic)
            q_next = self._get_qs_from_actions(obs_dict=batch["obs"], actions=cql_next_actions.permute(1, 0, 2), goal_dict=batch["goal_obs"], q_net=critic)
            q_cat = torch.cat([
                q_rand - cql_random_log_prob,
                q_next - cql_next_log_prob,
                q_curr - cql_curr_log_prob,
            ], dim=1)           # shape (B, 3 * N)
            q_cats.append(q_cat)

        # Calculate the losses for all critics
        cql_losses = []
        critic_losses = []
        cql_weight = torch.clamp(self.log_cql_weight.exp(), min=0.0, max=1000000.0)
        info["critic/cql_weight"] = cql_weight.item()
        for i, (q_pred, q_cat) in enumerate(zip(q_preds, q_cats)):
            # Calculate td error loss
            td_loss = self.td_loss_fcn(q_pred, q_target)
            # Calculate cql loss
            cql_loss = cql_weight * (self.min_q_weight * (torch.logsumexp(q_cat, dim=1).mean() - q_pred.mean()) -
                                     self.target_q_gap)
            cql_losses.append(cql_loss)
            # Calculate total loss
            loss = td_loss + cql_loss
            critic_losses.append(loss)
            info[f"critic/critic{i+1}_loss"] = loss

        # Run gradient descent if we're not validating
        if not validate:
            # Train CQL weight if tuning automatically
            if self.automatic_cql_tuning:
                cql_weight_loss = -torch.stack(cql_losses).mean()
                info[
                    "critic/cql_weight_loss"] = cql_weight_loss.item()  # Make sure to not store computation graph since we retain graph after backward() call
                self.optimizers["cql"].zero_grad()
                cql_weight_loss.backward(retain_graph=True)
                self.optimizers["cql"].step()
                info["critic/cql_grad_norms"] = self.log_cql_weight.grad.data.norm(2).pow(2).item()

            # Train critics
            for i, (critic_loss, critic, critic_target, optimizer) in enumerate(zip(
                    critic_losses, self.nets["critic"], self.nets["critic_target"], self.optimizers["critic"]
            )):
                retain_graph = (i < (len(critic_losses) - 1))
                critic_grad_norms = TorchUtils.backprop_for_loss(
                    net=critic,
                    optim=optimizer,
                    loss=critic_loss,
                    max_grad_norm=self.algo_config.critic.max_gradient_norm,
                    retain_graph=retain_graph,
                )
                info[f"critic/critic{i+1}_grad_norms"] = critic_grad_norms
                with torch.no_grad():
                    TorchUtils.soft_update(source=critic, target=critic_target, tau=self.algo_config.target_tau)

        # Return stats
        return info

    def _get_actions_and_log_prob(self, dist, sample_shape=torch.Size()):
        """
        Helper method to sample actions and compute corresponding log probabilities

        Args:
            dist (Distribution): Distribution to sample from
            sample_shape (torch.Size or tuple): Shape of output when sampling (number of samples)

        Returns:
            2-tuple:
                - (tensor) sampled actions (..., B, ..., A)
                - (tensor) corresponding log probabilities (..., B, ..., 1)
        """
        # Process networks with tanh differently than normal distributions
        if self.algo_config.actor.net.common.use_tanh:
            actions, actions_pre_tanh = dist.rsample(sample_shape=sample_shape, return_pretanh_value=True)
            log_prob = dist.log_prob(actions, pre_tanh_value=actions_pre_tanh).unsqueeze(dim=-1)
        else:
            actions = dist.rsample(sample_shape=sample_shape)
            log_prob = dist.log_prob(actions)

        return actions, log_prob

    @staticmethod
    def _get_qs_from_actions(obs_dict, actions, goal_dict, q_net):
        """
        Helper function for grabbing Q values given a single state and multiple (N) sampled actions.

        Args:
            obs_dict (dict): Observation dict from batch
            actions (tensor): Torch tensor, with dim1 assumed to be the extra sampled dimension
            goal_dict (dict): Goal dict from batch
            q_net (nn.Module): Q net to pass the observations and actions

        Returns:
            tensor: (B, N) corresponding Q values
        """
        # Get the number of sampled actions
        B, N, D = actions.shape

        # Repeat obs and goals in the batch dimension
        obs_dict_stacked = ObsUtils.repeat_and_stack_observation(obs_dict, N)
        goal_dict_stacked = ObsUtils.repeat_and_stack_observation(goal_dict, N)

        # Pass the obs and (flattened) actions through to get the Q values
        qs = q_net(obs_dict=obs_dict_stacked, acts=actions.reshape(-1, D), goal_dict=goal_dict_stacked)

        # Unflatten output
        qs = qs.reshape(B, N)

        return qs

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
        if "critic/cql_weight_loss" in info:
            loss_log["Critic/CQL_Weight"] = info["critic/cql_weight"]
            loss_log["Critic/CQL_Weight_Loss"] = info["critic/cql_weight_loss"]
            loss_log["Critic/CQL_Grad_Norms"] = info["critic/cql_grad_norms"]
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
        loss_log["Entropy_Weight_Loss"] = info["entropy_weight_loss"]
        loss_log["Entropy_Weight"] = info["entropy_weight"]
        if "entropy_grad_norms" in info:
            loss_log["Entropy_Grad_Norms"] = info["entropy_grad_norms"]
        return loss_log

    def set_train(self):
        """
        Prepare networks for evaluation. Update from super class to make sure
        target networks stay in evaluation mode all the time.
        """
        self.nets.train()

        # target networks always in eval
        for critic in self.nets["critic_target"]:
            critic.eval()

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
