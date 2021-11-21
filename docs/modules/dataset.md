# SequenceDataset

The `robomimic.utils.dataset.SequenceDataset` class extends PyTorch's default `torch.utils.data.Dataset` to interface with our demonstration [datasets](../introduction/datasets.html). The class supports accessing demonstration sub-sequences (as opposed to individual states) by index and both on-demand fetching and in-memory caching. This page walks through the key concepts of the `SequenceDataset` interface. Please refer to the official PyTorch [documentation](https://pytorch.org/docs/stable/data.html) and a short [example](../introduction/examples.html#train-loop-example) on how to use the `Dataset` and `DataLoader` interfaces to build a training pipeline. Also, please see the [released datasets section](../introduction/results.html#downloading-released-datasets) for guidance on downloading datasets that work out of the box with the repository.


Here is a sample dataset object:

```python
dataset = SequenceDataset(
    hdf5_path=dataset_path,
    obs_keys=(                      # observations we want to appear in batches
        "robot0_eef_pos", 
        "robot0_eef_quat", 
        "image", 
        "object",
    ),
    dataset_keys=(                  # can optionally specify more keys here if they should appear in batches
        "actions", 
        "rewards", 
        "dones",
    ),
    seq_length=10,                  # length-10 temporal sequences
    pad_seq_length=True,            # pad last obs per trajectory to ensure all sequences are sampled
    hdf5_cache_mode="all",          # cache dataset in memory to avoid repeated file i/o
    hdf5_normalize_obs=False,
    filter_by_attribute=None,       # can optionally provide a filter key here
)
```

- `hdf5_path`
	- The absolute / relative path to the hdf5 file containing training demonstrations. See [datasets](../introduction/datasets.html) page for the expected data structure.
- `obs_keys`
	- A list of strings specifying which observation modalities to read from the dataset. This is typically read from the config file: our implementation pools observation keys from `config.observation.modalities.obs.low_dim` and `config.observation.modalities.obs.rgb`.
- `dataset_keys`
	- Keys of non-observation data to read from a demonstration. Typically include `actions`, `rewards`, `dones`.
- `seq_length`
	- Length of the demonstration sub-sequence to fetch. 
- `pad_seq_length`
	- Whether to allow fetching subsequence that ends beyond the sequence. For example, given a demo of length 10 and `seq_length=10`, setting `pad_seq_length=True` allows the dataset object to access subsequence at `__get_item(index=5)__` by repeating the last frame 5 times.
- `hdf5_cache_mode`
	- Optionally cache the dataset in memory for faster access. The dataset supports three caching modes: `["all", "low_dim", or None]`. 
		- `all`: Load the entire dataset into the RAM. This mode minimizes data loading time but incurs the largest memory footprint. Recommended if the dataset is small or when working with low-dimensional observation data.
		- `low_dim`: Load only the low-dimensional observations into RAM. Always use this mode when possible as loading low-dim data incurs nontrivial overhead. Low-dim observations are specified at `config.observation.modalities.obs.low_dim`.
		- `None`: Always fetch data on-demand. 
- `hdf5_normalize_obs`
	- If `True`, normalize observations by computing the mean observation and std of each observation (in each dimension and modality), and normalizing unit mean and variance in each dimension.
- `filter_by_attribute`
  - if provided, use the provided filter key to look up a subset of demonstrations to load. See the documentation on [filter keys](../introduction/datasets.html#filter-keys-and-train-valid-splits) for more information.
