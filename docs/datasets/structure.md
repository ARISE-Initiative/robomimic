# Expected Structure of Datasets

TODO: add information on metadata (e.g. env_args)
TODO: consider a better representation of the dataset structure below, like maybe a highlighted box

The repository expects hdf5 datasets with a certain structure. The structure is shown below.

- data (group)

  - `total` (attribute) - number of state-action samples in the dataset

  - `env_args` (attribute) - a json string that contains metadata on the environment and relevant arguments used for collecting data

  - `mask` (group) - this group will exist in hdf5 datasets that contain filter keys

    - `<filter_key_1>` (dataset) - the first filter key. Note that the name of this dataset and length will vary. As an example, this could be the "valid" filter key, and contain the list ["demo_0", "demo_19", "demo_35"], corresponding to 3 validation trajectories.

  - `demo_0` (group) - group for the first trajectory (every trajectory has a group)

    - `num_samples` (attribute) - the number of state-action samples in this trajectory

    - `model_file` (attribute) - the xml string corresponding to the MJCF MuJoCo model. Only present for robosuite datasets.

    - `states` (dataset) - flattened raw MuJoCo states, ordered by time. Shape (N, D) where N is the length of the trajectory, and D is the dimension of the state vector. Should be empty or have dummy values for non-robosuite datasets.

    - `actions` (dataset) - environment actions, ordered by time. Shape (N, A) where N is the length of the trajectory, and A is the action space dimension

    - `rewards` (dataset) - environment rewards, ordered by time. Shape (N,) where N is the length of the trajectory.

    - `dones` (dataset) - done signal, equal to 1 if playing the corresponding action in the state should terminate the episode. Shape (N,) where N is the length of the trajectory.

    - `obs` (group) - group for the observation keys. Each key is stored as a dataset.

      - `<obs_key_1>` (dataset) - the first observation key. Note that the name of this dataset and shape will vary. As an example, the name could be "agentview_image", and the shape could be (N, 84, 84, 3). 

        ...

    - `next_obs` (group) - group for the next observations.

      - `<obs_key_1>` (dataset) - the first observation key.

        ...

  - `demo_1` (group) - group for the second trajectory

    ...

## Storing image observations

The repository expects image observations stored in the hdf5 to be of type `np.uint8` and be stored in channel-last `(H, W, C)` format. This is for two reasons - (1) this is a common format that many `gym` environments and all `robosuite` environments return image observations in, and (2) using `np.uint8` saves space in dataset storage, as opposed to using floats. Note that the robosuite observation extraction script (`dataset_states_to_obs.py`) already stores images in the correct format.

## Storing actions

The repository **expects all actions to be normalized** between -1 and 1 (this makes for easier policy learning and allows the use of `tanh` layers). The `get_dataset_info.py` script can be used to sanity check the actions in a dataset, as it will throw an `Exception` if there is a violation.

## Filter Keys and Train-Valid Splits

Each filter key is a dataset in the "mask" group of the dataset hdf5, which contains a list of the demo group keys - these correspond to subsets of trajectories in the dataset. Filter keys make it easy to train on a subset of the data present in an hdf5. A common use is to split a dataset into training and validation datasets using the `split_train_val.py` script. 

```sh
$ python split_train_val.py --dataset /path/to/dataset.hdf5 --ratio 0.1
```

The example above creates a `train` filter key and a `valid` filter key under `mask/train` and `mask/valid`, where the former contains a list of demo groups corresponding to a 90% subset of the dataset trajectories, and the latter contains a list of demo groups correspond to a 10% subset of the dataset trajectories. These filter keys are used by the data loader during training if `config.experiment.validate` is set to True in the training config. 

Many of the released datasets contain other filter keys besides the train-val splits. Some contain `20_percent` and `50_percent` filter keys corresponding to data subsets, and the Multi-Human datasets contain filter keys that correspond to each operator's data (e.g. `better_operator_1`, `better_operator_2`), and ones that correspond to different combinations of operator data (e.g. `better`, `worse_better`). 

Using these filter keys during training is simple. For example, to use the `20_percent` subset during training, you can simply set `config.train.hdf5_filter_key = "20_percent"` in the training config. If using validation, then the `20_percent_train` and `20_percent_valid` filter keys will also be used -- these were generated using the `split_train_val.py` script by passing `--filter_key 20_percent`.

For robosuite datasets, if attempting to create your own train-val splits, we recommend running the `split_train_val.py` script on the `demo.hdf5` file before extracting observations, since filter keys are copied from the source hdf5 during observation extraction (see more details below on robosuite hdf5s). This will ensure that all postprocessed hdf5s generated from the `demo.hdf5` inherits the same filter keys.