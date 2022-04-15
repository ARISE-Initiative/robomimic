# Working with robomimic Modules

This section discusses some simple examples packaged with the repository (in the top-level `examples` folder) that provide a more thorough understanding of components used in the repository. These examples are meant to assist users who may want to build on these components, or use these components in other applications, in contrast to the [Getting Started](./quickstart.html) section, which provides examples relevant to using the repository as-is.

## Train Loop Example

We include a simple example script in `examples/simple_train_loop.py` to show how easy it is to use our `SequenceDataset` class and standardized hdf5 datasets in a general torch training loop. Run the example using the command below.

```sh
$ python examples/simple_train_loop.py
```

Modifying this example for use in other code repositories is simple. First, create the dataset loader as in the script.

```python
from robomimic.utils.dataset import SequenceDataset

def get_data_loader(dataset_path):
    """
    Get a data loader to sample batches of data.
    """
    dataset = SequenceDataset(
        hdf5_path=dataset_path,
        obs_keys=(                      # observations we want to appear in batches
            "robot0_eef_pos", 
            "robot0_eef_quat", 
            "robot0_gripper_qpos", 
            "object",
        ),
        dataset_keys=(                  # can optionally specify more keys here if they should appear in batches
            "actions", 
            "rewards", 
            "dones",
        ),
        load_next_obs=True,
        frame_stack=1,
        seq_length=10,                  # length-10 temporal sequences
        pad_frame_stack=True,
        pad_seq_length=True,            # pad last obs per trajectory to ensure all sequences are sampled
        get_pad_mask=False,
        goal_mode=None,
        hdf5_cache_mode="all",          # cache dataset in memory to avoid repeated file i/o
        hdf5_use_swmr=True,
        hdf5_normalize_obs=False,
        filter_by_attribute=None,       # can optionally provide a filter key here
    )
    print("\n============= Created Dataset =============")
    print(dataset)
    print("")

    data_loader = DataLoader(
        dataset=dataset,
        sampler=None,       # no custom sampling logic (uniform sampling)
        batch_size=100,     # batches of size 100
        shuffle=True,
        num_workers=0,
        drop_last=True      # don't provide last batch in dataset pass if it's less than 100 in size
    )
    return data_loader

data_loader = get_data_loader(dataset_path="/path/to/your/dataset.hdf5")
```

Then, construct your model, and use the same pattern as in the `run_train_loop` function in the script, to iterate over batches to train the model.

```python
for epoch in range(1, num_epochs + 1):
  
    # iterator for data_loader - it yields batches
    data_loader_iter = iter(data_loader)
    
    for train_step in range(gradient_steps_per_epoch):
        # load next batch from data loader
        try:
            batch = next(data_loader_iter)
        except StopIteration:
            # data loader ran out of batches - reset and yield first batch
            data_loader_iter = iter(data_loader)
            batch = next(data_loader_iter)

        # @batch is a dictionary with keys loaded from the dataset.
        # Train your model on the batch below.
    
```



## Config Example

The simple config example script at `examples/simple_config.py` shows how the `Config` object can easily be instantiated and modified safely with different levels of locking. We reproduce certain portions of the script. First, we can create a `Config` object and call `lock` when we think we won't need to change it anymore.

```python
from robomimic.config.base_config import Config

# create config
config = Config()

# add nested attributes to the config
config.train.batch_size = 100
config.train.learning_rate = 1e-3
config.algo.actor_network_size = [1000, 1000]
config.lock()  # prevent accidental changes
```

Now, when we try to add a new key (or modify the value of an existing key), the config will throw an error.

```python
# the config is locked --- cannot add new keys or modify existing keys
try:
    config.train.optimizer = "Adam"
except RuntimeError as e:
    print(e)
```

However, the config can be safely modified using appropriate contexts.

```python
# values_unlocked scope allows modifying values of existing keys, but not adding keys
with config.values_unlocked():
    config.train.batch_size = 200
print("batch_size={}".format(config.train.batch_size))

# unlock config within the scope, allowing new keys to be inserted
with config.unlocked():
    config.test.num_eval = 10

# verify that the config remains locked outside of the scope
assert config.is_locked
assert config.test.is_locked
```

Finally, the config can also be updated by using external dictionaries - this is helpful for loading config jsons.

```python
# update this config with external config from a dict
ext_config = {
    "train": {
        "learning_rate": 1e-3
    },
    "algo": {
        "actor_network_size": [1000, 1000]
    }
}
with config.values_unlocked():
    config.update(ext_config)

print(config)
```

Please see the [Config documentation](../modules/configs.html) for more information on Config objects.



## Observation Networks Example

The example script in `examples/simple_obs_net.py` discusses how to construct networks for taking observation dictionaries as input, and that produce dictionaries as outputs. See [this section](../modules/models.html#observation-encoder-and-decoder) in the documentation for more details.



## Custom Observation Modalities Example

The example script in `examples/add_new_modality.py` discusses how to (a) modify pre-existing observation modalities, and (b) add your own custom observation modalities with custom encoding. See [this section](../modules/models.html#observation-encoder-and-decoder) in the documentation for more details about the encoding and decoding process.