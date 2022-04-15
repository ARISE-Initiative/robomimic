# Algorithms

The `Algo` class is an abstraction used to make creating and training networks easy.

### Initialization

The standard entry point for creating `Algo` class instances is the `algo_factory` function in `algo/algo.py`. This uses a mapping from an algo name (e.g. `"bc"`) to a special `algo_config_to_class` function, that is responsible for reading an `algo_config` (`config.algo` section of the config) and returning the appropriate algo class name to instantiate, along with any additional keyword arguments needed. This is necessary because algorithms can actually have multiple subclasses with different functionality - for example, BC has the `BC_GMM` class for training GMM policies, and the `BC_RNN` class for training RNN policies. 

Therefore, every algorithm file (for example `algo/bc.py`) implements an `algo_config_to_class` function. The function should have the `register_algo_factory_func` decorator with the algo name (e.g. `"bc"`) - this registers the function into the registry used by `algo_factory`. The algo name should match the `ALGO_NAME` property in the corresponding config class for the algorithm (for BC, this is in the `BCConfig` class in `configs/bc_config.py`). The implementation from `algo/bc.py` is reproduced below.

```python
@register_algo_factory_func("bc")
def algo_config_to_class(algo_config):
    """
    Maps algo config to the BC algo class to instantiate, along with additional algo kwargs.

    Args:
        algo_config (Config instance): algo config

    Returns:
        algo_class: subclass of Algo
        algo_kwargs (dict): dictionary of additional kwargs to pass to algorithm
    """

    # note: we need the check below because some configs import BCConfig and exclude
    # some of these options
    gaussian_enabled = ("gaussian" in algo_config and algo_config.gaussian.enabled)
    gmm_enabled = ("gmm" in algo_config and algo_config.gmm.enabled)
    vae_enabled = ("vae" in algo_config and algo_config.vae.enabled)

    if algo_config.rnn.enabled:
        if gmm_enabled:
            return BC_RNN_GMM, {}
        return BC_RNN, {}
    assert sum([gaussian_enabled, gmm_enabled, vae_enabled]) <= 1
    if gaussian_enabled:
        return BC_Gaussian, {}
    if gmm_enabled:
        return BC_GMM, {}
    if vae_enabled:
        return BC_VAE, {}
    return BC, {}
```



### Important Class Methods

In this section, we outline important class methods that each `Algo` subclass needs to implement or override, categorizing them by whether they are usually called during initialization, at train time, or test time.

- **Initialization**
  - `_create_networks(self)`
    - Called on class initialization - should construct networks and place them into the `self.nets` ModuleDict
- **Train**
  - `process_batch_for_training(self, batch)`
    - Takes a batch sampled from the data loader, and filters out the relevant portions needed for the algorithm. It should also send the batch to the correct device (cpu or gpu).
  - `train_on_batch(self, batch, epoch, validate=False)`
    - Takes a processed batch, and trains all networks on the batch of data, taking the epoch number and whether this is a training or validation batch into account. This is where the main logic for training happens (e.g. forward and backward passes for networks). Should return a dictionary of important training statistics (e.g. loss on the batch, gradient norms, etc.)
  - `log_info(self, info)`
    - Takes the output of `train_on_batch` and returns a new processed dictionary for tensorboard logging.
  - `set_train(self)`
    - Prepares network modules for training. By default, just calls `self.nets.train()`, but certain algorithms may always want a subset of the networks in evaluation mode (such as target networks for BCQ). In this case they should override this method.
  - `on_epoch_end(self, epoch)`
    - Called at the end of each training epoch. Usually consists of stepping learning rate schedulers (if they are being used).
  - `serialize(self)`
    - Returns the state dictionary that contains the current model parameters. This is used to produce agent checkpoints. By default, returns `self.nets.state_dict()` - usually only needs to be overriden by hierarchical algorithms like HBC and IRIS to collect state dictionaries from sub-algorithms.
- **Test**
  - `set_eval(self)`
    - Prepares network modules for evaluation. By default, just calls `self.nets.eval()`, but certain hierarchical algorithms like HBC and IRIS override this to call `set_eval` on their sub-algorithms.
  - `deserialize(self, model_dict)`
    - Inverse operation of `serialize` - load model weights. Used at test-time to restore model weights.
  - `get_action(self, obs_dict, goal_dict=None)`
    - The primary method that is called at test-time to return one or more actions, given observations. 
  - `reset(self)`
    - Called at the beginning of each rollout episode to clear internal agent state before starting a rollout. As an example, `BC_RNN` resets the step counter and hidden state.

### Train Loop

We reproduce the stripped down version of the train loop from `examples/simple_train_loop.py` to show how methods of `Algo` instances are used during training.

```python
# @model should be instance of Algo class to use for training
# @data_loader should be instance of torch.utils.data.DataLoader for sampling batches

# train for 50 epochs and 100 gradient steps per epoch
num_epochs = 50
gradient_steps_per_epoch = 100

# ensure model is in train mode
model.set_train()

for epoch in range(1, num_epochs + 1): # epoch numbers start at 1
    # iterator for data_loader - it yields batches
    data_loader_iter = iter(data_loader)

    # record losses
    losses = []

    for _ in range(gradient_steps_per_epoch):

        # load next batch from data loader
        try:
            batch = next(data_loader_iter)
        except StopIteration:
            # data loader ran out of batches - reset and yield first batch
            data_loader_iter = iter(data_loader)
            batch = next(data_loader_iter)

        # process batch for training
        input_batch = model.process_batch_for_training(batch)

        # forward and backward pass
        info = model.train_on_batch(batch=input_batch, epoch=epoch, validate=False)

        # record loss
        step_log = model.log_info(info)
        losses.append(step_log["Loss"])

    # save model
    model_params = model.serialize()
    model_dict = dict(model=model.serialize())
    torch.save(model_dict, /path/to/ckpt.pth)
        
    # do anything model needs to after finishing epoch
    model.on_epoch_end(epoch)
```



### Test Time

We reproduce some logic from the `policy_from_checkpoint` function defined in `utils/file_utils.py` to show how `Algo` methods are used to load a model.

```python
# load checkpoint
ckpt_dict = maybe_dict_from_checkpoint(ckpt_path=/path/to/ckpt.pth)
algo_name = ckpt_dict["algo_name"]
config, _ = config_from_checkpoint(algo_name=algo_name, ckpt_dict=ckpt_dict)

# create Algo instance
model = algo_factory(
    algo_name,
    config,
    obs_key_shapes=ckpt_dict["shape_metadata"]["all_shapes"],
    ac_dim=ckpt_dict["shape_metadata"]["ac_dim"],
    device=device,
)

# load weights
model.deserialize(ckpt_dict["model"])
model.set_eval()

# rollout wrapper around model
model = RolloutPolicy(model)
```

We also reproduce a rollout loop to show how the `RolloutPolicy` wrapper (see `algo/algo.py`) is used to easily deploy trained models in the environment.

```python
# @policy should be instance of RolloutPolicy
assert isinstance(policy, RolloutPolicy)

# episode reset (calls @set_eval and @reset)
policy.start_episode()
obs = env.reset()

horizon = 400
total_return = 0
for step_i in range(horizon):
    # get action from policy (calls @get_action)
    act = policy(obs)
    # play action
    next_obs, r, done = env.step(act)
    total_return += r
    success = env.is_success()["task"]
    if done or success:
        break
```



## Implemented Algorithms

### BC

- Vanilla Behavioral Cloning (see [this paper](https://papers.nips.cc/paper/1988/file/812b4ba287f5ee0bc9d43bbf5bbe87fb-Paper.pdf)), consisting of simple supervised regression from observations to actions. Implemented in the `BC` class in `algo/bc.py`, along with some variants such as `BC_GMM` (stochastic GMM policy) and `BC_VAE` (stochastic VAE policy)

### BC-RNN

- Behavioral Cloning with an RNN network. Implemented in the `BC_RNN` and `BC_RNN_GMM` (recurrent GMM policy) classes in `algo/bc.py`.

### HBC

- Hierarchical Behavioral Cloning - the implementation is largely based off of [this paper](https://arxiv.org/abs/2003.06085). Implemented in the `HBC` class in `algo/hbc.py`.

### IRIS

- A recent batch offline RL algorithm from [this paper](https://arxiv.org/abs/1911.05321). Implemented in the `IRIS` class in `algo/iris.py`.

### BCQ

- A recent batch offline RL algorithm from [this paper](https://arxiv.org/abs/1812.02900). Implemented in the `BCQ` class in `algo/bcq.py`.

### CQL

- A recent batch offline RL algorithm from [this paper](https://arxiv.org/abs/2006.04779). Implemented in the `CQL` class in `algo/cql.py`.

### TD3-BC

- A recent algorithm from [this paper](https://arxiv.org/abs/2106.06860). We implemented it as an example (see section below on building your own algorithm). Implemented in the `TD3_BC` class in `algo/td3_bc.py`.



## Building your own Algorithm

In this section, we walk through an example of implementing a custom algorithm, to show how easy it is to extend the functionality in the repository. We choose to implement the recently proposed [TD3-BC](https://arxiv.org/abs/2106.06860) algorithm. 

This requires implementing two new files - `algo/td3_bc.py` (which contains the `Algo` subclass implementation) and `config/td3_bc_config.py` (which contains the `Config` subclass implementation). We also make sure to add the line `from robomimic.algo.td3_bc import TD3_BC` to `algo/__init__.py` and `from robomimicL.config.td3_bc_config import TD3_BCConfig` to `config/__init__.py` to `config/__init__.py` to make sure the `Algo` and `Config` subclasses can be found.

We first describe the config implementation - we implement a `TD3_BCConfig` config class that subclasses from `BaseConfig`. Importantly, we set the class variable `ALGO_NAME = "td3_bc"` to register this config under that algo name. We implement the `algo_config` function to populate `config.algo` with the keys needed for the algorithm - it is extremely similar to the `BCQConfig` implementation. Portions of the code are reproduced below.

```python
class TD3_BCConfig(BaseConfig):
    ALGO_NAME = "td3_bc"
    
    def algo_config(self):
        # optimization parameters
        self.algo.optim_params.critic.learning_rate.initial = 3e-4      # critic learning rate
        self.algo.optim_params.critic.learning_rate.decay_factor = 0.1  # factor to decay LR by (if epoch schedule non-empty)
        self.algo.optim_params.critic.learning_rate.epoch_schedule = [] # epochs where LR decay occurs
        self.algo.optim_params.critic.regularization.L2 = 0.00          # L2 regularization strength
        self.algo.optim_params.critic.start_epoch = -1                  # number of epochs before starting critic training (-1 means start right away)
        self.algo.optim_params.critic.end_epoch = -1                    # number of epochs before ending critic training (-1 means start right away)

        self.algo.optim_params.actor.learning_rate.initial = 3e-4       # actor learning rate
        self.algo.optim_params.actor.learning_rate.decay_factor = 0.1   # factor to decay LR by (if epoch schedule non-empty)
        self.algo.optim_params.actor.learning_rate.epoch_schedule = []  # epochs where LR decay occurs
        self.algo.optim_params.actor.regularization.L2 = 0.00           # L2 regularization strength
        self.algo.optim_params.actor.start_epoch = -1                   # number of epochs before starting actor training (-1 means start right away)
        self.algo.optim_params.actor.end_epoch = -1                     # number of epochs before ending actor training (-1 means start right away)

        # alpha value - for weighting critic loss vs. BC loss
        self.algo.alpha = 2.5

        # target network related parameters
        self.algo.discount = 0.99                       # discount factor to use
        self.algo.n_step = 1                            # for using n-step returns in TD-updates
        self.algo.target_tau = 0.005                    # update rate for target networks
        self.algo.infinite_horizon = False              # if True, scale terminal rewards by 1 / (1 - discount) to treat as infinite horizon

        # ================== Critic Network Config ===================
        self.algo.critic.use_huber = False              # Huber Loss instead of L2 for critic
        self.algo.critic.max_gradient_norm = None       # L2 gradient clipping for critic (None to use no clipping)
        self.algo.critic.value_bounds = None            # optional 2-tuple to ensure lower and upper bound on value estimates 

        # critic ensemble parameters (TD3 trick)
        self.algo.critic.ensemble.n = 2                 # number of Q networks in the ensemble
        self.algo.critic.ensemble.weight = 1.0          # weighting for mixing min and max for target Q value

        self.algo.critic.layer_dims = (256, 256, 256)   # size of critic MLP

        # ================== Actor Network Config ===================

        # update actor and target networks every n gradients steps for each critic gradient step
        self.algo.actor.update_freq = 2

        # exploration noise used to form target action for Q-update - clipped Gaussian noise
        self.algo.actor.noise_std = 0.2                 # zero-mean gaussian noise with this std is applied to actions
        self.algo.actor.noise_clip = 0.5                # noise is clipped in each dimension to (-noise_clip, noise_clip)

        self.algo.actor.layer_dims = (256, 256, 256)    # size of actor MLP
```

Usually, we only need to implement the `algo_config` function to populate `config.algo` with the keys needed for the algorithm, but we also update the `experiment_config` function and `observation_config` function to make it easier to reproduce experiments on `gym` environments from the paper. See the source file for more details.

Now we discuss the algorithm implementation. As described in the "Initialization" section above, we first need to implement the `algo_config_to_class` method - this is straightforward since we don't have multiple variants of this algorithm. We take special care to make sure we register this function with the same algo name that we used for defining the config (`"td3_bc"`).

```python
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
```

Next, we'll describe how we implement the methods outlined in the "Important Methods" section above. We omit several of the methods, since their implementation is extremely similar to the `BCQ` implementation. We start by defining the class and implementing `_create_networks`. The code uses helper functions `_create_critics` and `_create_actor` to create the critic and actor networks, as in the `BCQ` implementation.

```python
class TD3_BC(PolicyAlgo, ValueAlgo):
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
```

Next we describe the `train_on_batch` function, which implements the main training logic. The function trains the critic using the `_train_critic_on_batch` helper function, and then actor using the `_train_actor_on_batch` helper function (the actor is trained at a slower rate according to the `config.algo.actor.update_freq` config variable, as in the original author's implementation). Finally, the target network parameters are moved a little closer to the current network parameters, using `TorchUtils.soft_update`.

```python
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
```

Below, we show the helper functions for training the critics, to be explicit in how the Bellman backup is used to construct the TD loss. The target Q values for the TD loss are obtained in the same way as [TD3](https://arxiv.org/abs/1802.09477).

```python
    def _train_critic_on_batch(self, batch, epoch, no_backprop=False):
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
        
    def _get_target_values(self, next_states, goal_states, rewards, dones):
        """
        Helper function to get target values for training Q-function with TD-loss.
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
        """
        q_estimated = critic(states, actions, goal_states)
        if self.algo_config.critic.use_huber:
            critic_loss = nn.SmoothL1Loss()(q_estimated, q_targets)
        else:
            critic_loss = nn.MSELoss()(q_estimated, q_targets)
        return critic_loss
```

Next we show the helper function for training the actor, which is trained through a weighted combination of the TD3 (DDPG) and BC loss.

```python
    def _train_actor_on_batch(self, batch, epoch, no_backprop=False):
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
```

Finally, we describe the `get_action` implementation - which is used at test-time during rollouts. The implementation is extremely simple - just query the actor for an action.

```python
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
```

That's it! See `algo/td3_bc.py` for the complete implementation, and compare it to `algo/bcq.py` to see the similarity between the two implementations. 

We can now run the `generate_config_templates.py` script to generate the json template for our new algorithm, and then run it on our desired dataset.

```sh
# generate ../exps/templates/td3_bc.json
$ python generate_config_templates.py 

# run training
$ python train.py --config ../exps/templates/td3_bc.json --dataset /path/to/walker2d_medium_expert.hdf5
```

