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

Refer [here](../introduction/implemented_algorithms.html) for the list of algorithms currently implemented in robomimic



## Building your own Algorithm

Learn how to implement your own learning algorithm [here](../tutorials/custom_algorithms.html)
