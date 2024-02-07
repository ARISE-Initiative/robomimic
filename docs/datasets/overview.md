# Overview

## Dataset Pipeline

<style>
table{
margin: auto;
}
</style>

Datasets capture recorded environment data and are used as inputs to a given offline RL or IL algorithm in **robomimic**. In general, you can use datasets with **robomimic** by:

1. **Downloading** the desired dataset
2. **Postprocessing** the dataset to guarantee compatibility with robomimic
3. **Training** agent(s) in robomimic with dataset

**robomimic** currently supports the following datasets out of the box. Click on the corresponding **(1) Downloading** link to download the dataset and the corresponding **(2) Postprocessing** link for postprocessing that dataset.


|          **Dataset**          | <center>**Task Types**</center> | **Downloading** | **Postprocessing**  |
| ----------------------------- | :-------------: | :-------------: | :-------------: |
| [**robomimic v0.1**](robomimic_v0.1.html)| Sim + Real Robot Manipulation | [Link](robomimic_v0.1.html#downloading)  | [Link](robomimic_v0.1.html#postprocessing)  |
| [**MimicGen**](mimicgen.html)              | Sim Robot Manipulation | [Link](mimicgen.html#downloading)  | [Link](mimicgen.html#postprocessing)  |
| [**D4RL**](d4rl.html)                      | Sim Locomotion | [Link](d4rl.html#downloading)  | [Link](d4rl.html#postprocessing)  |
| [**MOMART**](momart.html)                    | Sim Mobile Manipulation | [Link](momart.html#downloading)  | [Link](momart.html#postprocessing)  |
| [**RoboTurk Pilot**](roboturk_pilot.html)            | Sim Robot Manipulation | [Link](roboturk_pilot.html#downloading)  | [Link](roboturk_pilot.html#postprocessing)  |


After downloading and postprocessing, **(3) Training** with the dataset is straightforward and unified across all datasets:

```sh
python train.py --dataset <PATH_TO_POSTPROCESSED_DATASET> --config <PATH_TO_CONFIG>
```

## Generating Your Own Dataset

**robomimic** provides tutorials for collecting custom datasets for specific environment platforms. Click on any of the links below for more information for the specific environment setup:

|          **Environment Platform**          | **Task Types** |
| ----------------------------- | :---------------------: |
| [**robosuite**](robosuite.html)| Robot Manipulation  |

<div class="admonition note">
<p class="admonition-title">Create Your Own Environment Wrapper!</p>

If you want to generate your own dataset in a custom environment platform that is not listed above, please see [this page](../modules/environments.md#implement-an-environment-wrapper).

</div>


## Dataset Structure

All postprocessed **robomimic** compatible datasets share the same data structure. A single dataset is a single HDF5 file with the following structure:

<details>
  <summary><b>HDF5 Structure <span style="color:red;">(click to expand)</span></b></summary>
<p>

- **`data`** (group)

  - **`total`** (attribute) - number of state-action samples in the dataset

  - **`env_args`** (attribute) - a json string that contains metadata on the environment and relevant arguments used for collecting data. Three keys: `env_name`, the name of the environment or task to create, `env_type`, one of robomimic's supported [environment types](https://github.com/ARISE-Initiative/robomimic/blob/master/robomimic/envs/env_base.py#L9), and `env_kwargs`, a dictionary of keyword-arguments to be passed into the environment of type `env_name`.

  - **`demo_0`** (group) - group for the first trajectory (every trajectory has a group)

    - **`num_samples`** (attribute) - the number of state-action samples in this trajectory

    - **`model_file`** (attribute) - the xml string corresponding to the MJCF MuJoCo model. Only present for robosuite datasets.

    - **`states`** (dataset) - flattened raw MuJoCo states, ordered by time. Shape (N, D) where N is the length of the trajectory, and D is the dimension of the state vector. Should be empty or have dummy values for non-robosuite datasets.

    - **`actions`** (dataset) - environment actions, ordered by time. Shape (N, A) where N is the length of the trajectory, and A is the action space dimension

    - **`rewards`** (dataset) - environment rewards, ordered by time. Shape (N,) where N is the length of the trajectory.

    - **`dones`** (dataset) - done signal, equal to 1 if playing the corresponding action in the state should terminate the episode. Shape (N,) where N is the length of the trajectory.

    - **`obs`** (group) - group for the observation keys. Each key is stored as a dataset.

      - **`<obs_key_1>`** (dataset) - the first observation key. Note that the name of this dataset and shape will vary. As an example, the name could be "agentview_image", and the shape could be (N, 84, 84, 3). 

        ...

    - **`next_obs`** (group) - group for the next observations.

      - **`<obs_key_1>`** (dataset) - the first observation key.

        ...

  - **`demo_1`** (group) - group for the second trajectory

    ...
    
- **`mask`** (group) - this group will exist in hdf5 datasets that contain filter keys

  - **`<filter_key_1>`** (dataset) - the first filter key. Note that the name of this dataset and length will vary. As an example, this could be the "valid" filter key, and contain the list ["demo_0", "demo_19", "demo_35"], corresponding to 3 validation trajectories.

    ...

</p>
</details>

### Data Conventions

**robomimic**-compatible datasets expect certain values (such as images and actions) to be formatted a specific way. See the below sections for further details:

<details>
  <summary><b>Storing images</b></summary>
<p>
<div class="admonition warning">
<p class="admonition-title">Warning!</p>

Dataset images should be of type `np.uint8` and be stored in channel-last `(H, W, C)` format. This is because:

- **(1)** this is a common format that many `gym` environments and all `robosuite` environments return image observations in
- **(2)** using `np.uint8` (vs floats) saves space in dataset storage

Note that the robosuite observation extraction script (`dataset_states_to_obs.py`) already stores images in the correct format.

</div>

</p>
</details>


<details>
  <summary><b>Storing actions</b></summary>
<p>
<div class="admonition warning">
<p class="admonition-title">Warning!</p>

Actions should be **normalized between -1 and 1**. This is because this range enables easier policy learning via the use of `tanh` layers).

The `get_dataset_info.py` script can be used to sanity check stored actions, and will throw an `Exception` if there is a violation.

</div>

</p>
</details>

### Filter Keys

Filter keys enable arbitrary splitting of a dataset into sub-groups, and allow training on a specific subset of the data.

A common use-case is to split data into train-validation splits. We provide a convenience script for doing this in the `robomimic/scripts` directory:

```sh
$ python split_train_val.py --dataset /path/to/dataset.hdf5 --ratio 0.1 --filter_key <FILTER_KEY_NAME>
```

- `--dataset` specifies the path to the hdf5 dataset
- `--ratio` specifies the amount of validation data you would like to create. In the example above, 10% of the demonstrations will be put into the validation group.
- `--filter_key` (optional) By default, this script splits all demonstration keys in the hdf5 into 2 new hdf5 groups - one under `mask/train`, and one under `mask/valid`. If this argument is provided, the demonstration keys corresponding to this filter key (under `mask/<FILTER_KEY_NAME>`) will be split into 2 groups - `mask/<FILTER_KEY_NAME>_train` and `mask/<FILTER_KEY_NAME>_valid`.

<div class="admonition note">
<p class="admonition-title">Note!</p>

You can easily list the filter keys present in a dataset with the `get_dataset_info.py` script (see [this link](../tutorials/dataset_contents.html#view-dataset-structure-and-videos)), and you can even pass a `--verbose` flag to list the exact demonstrations that each filter key corresponds to.

</div>

Using filter keys during training is easy. To use the generated train-valid split, you can set `config.experiment.validate=True` to ensure that validation will run after each training epoch, and then set `config.train.hdf5_filter_key="train"` and `config.train.hdf5_validation_filter_key="valid"` so that the demos under `mask/train` are used for training, and the demos under `mask/valid` are used for validation. 

You can also use a custom filter key for training by setting `config.train.hdf5_filter_key=<FILTER_KEY_NAME>`. This ensures that only the demos under `mask/<FILTER_KEY_NAME>` are used during training. You can also specify a custom filter key for validation by setting `config.train.hdf5_validation_filter_key`.
