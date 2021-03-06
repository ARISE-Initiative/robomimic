# robosuite Datasets

The repository is fully compatible with datasets collected using [robosuite](https://robosuite.ai/). See [this link](https://robosuite.ai/docs/algorithms/demonstrations.html) for more information on collecting your own human demonstrations using robosuite. 

## Converting robosuite hdf5 datasets

The raw `demo.hdf5` file generated by the `collect_human_demonstrations.py` robosuite script can easily be modified in-place to be compatible with **robomimic**:

```sh
$ python conversion/convert_robosuite.py --dataset /path/to/demo.hdf5
```

<div class="admonition info">
<p class="admonition-title">Post-Processed Dataset Structure</p>

This post-processed `demo.hdf5` file in its current state is _missing_ observations (e.g.: proprioception, images, ...), rewards, and dones, which are necessary for training policies.

However, keeping these observation-free datasets is useful because it **allows flexibility in [extracting](robosuite.md#extracting-observations-from-mujoco-states) different kinds of observations and rewards**.

<details>
  <summary><b>Dataset Structure <span style="color:red;">(click to expand)</span></b></summary>
<p>

- `data` (group)

  - `total` (attribute) - number of state-action samples in the dataset

  - `env_args` (attribute) - a json string that contains metadata on the environment and relevant arguments used for collecting data

  - `demo_0` (group) - group for the first demonstration (every demonstration has a group)

    - `num_samples` (attribute) - the number of state-action samples in this trajectory
    - `model_file` (attribute) - the xml string corresponding to the MJCF MuJoCo model
    - `states` (dataset) - flattened raw MuJoCo states, ordered by time
    - `actions` (dataset) - environment actions, ordered by time

  - `demo_1` (group) - group for the second demonstration

    ...
</p>
</details>

</div>


Next, we will extract observations from this raw dataset.


## Extracting Observations from MuJoCo states

<div class="admonition warning">
<p class="admonition-title">Warning! Train-Validation Data Splits</p>

For robosuite datasets, if using your own [train-val splits](overview.md#filter-keys), generate these splits _before_ extracting observations. This ensures that all postprocessed hdf5s generated from the `demo.hdf5` inherits the same filter keys.

</div>

Generating observations from a dataset is straightforward and can be done with a single command from `robomimic/scripts`:

```sh
# For low dimensional observations only, with done on task success
$ python dataset_states_to_obs.py --dataset /path/to/demo.hdf5 --output_name low_dim.hdf5 --done_mode 2

# For including image observations
$ python dataset_states_to_obs.py --dataset /path/to/demo.hdf5 --output_name image.hdf5 --done_mode 2 --camera_names agentview robot0_eye_in_hand --camera_height 84 --camera_width 84

# Using dense rewards
$ python dataset_states_to_obs.py --dataset /path/to/demo.hdf5 --output_name image_dense.hdf5 --done_mode 2 --dense --camera_names agentview robot0_eye_in_hand --camera_height 84 --camera_width 84

# Only writing done at the end of the trajectory
$ python dataset_states_to_obs.py --dataset /path/to/demo.hdf5 --output_name image_done_1.hdf5 --done_mode 1 --camera_names agentview robot0_eye_in_hand --camera_height 84 --camera_width 84

# For seeing descriptions of all the command-line args available
$ python dataset_states_to_obs.py --help
```

## Citation
```sh
@article{zhu2020robosuite,
  title={robosuite: A modular simulation framework and benchmark for robot learning},
  author={Zhu, Yuke and Wong, Josiah and Mandlekar, Ajay and Mart{\'\i}n-Mart{\'\i}n, Roberto},
  journal={arXiv preprint arXiv:2009.12293},
  year={2020}
}
```