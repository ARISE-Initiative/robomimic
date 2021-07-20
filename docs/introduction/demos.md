# Demo Showcases

## Hyperparam Sweeps

- TODO

## Simple Train Loop

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

