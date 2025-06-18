# Dataset Contents and Visualization

This tutorial shows how to view contents of robomimic hdf5 datasets.

## Viewing HDF5 Dataset Structure

<div class="admonition note">
<p class="admonition-title">Note: HDF5 Dataset Structure.</p>

[This link](../datasets/overview.html#dataset-structure) shows the expected structure of each hdf5 dataset.

</div>

The repository offers a simple utility script (`get_dataset_info.py`) to view the hdf5 dataset structure and some statistics of hdf5 datasets. The script displays the following information:

- statistics about the trajectories (number, average length, etc.)
- the [filter keys](../datasets/overview.html#filter-keys) in the dataset
- the [environment metadata](../modules/environments.html#initialize-an-environment-from-a-dataset) in the dataset, which is used to construct the same simulator environment that the data was collected on
- the dataset structure for the first demonstration

Pass the `--verbose` argument to print the list of demonstration keys under each filter key, and the dataset structure for all demonstrations. An example, using the small hdf5 dataset packaged with the repository in `tests/assets/test_v15.hdf5` is shown below.

```sh
$ python get_dataset_info.py --dataset ../../tests/assets/test_v15.hdf5
```

<div class="admonition tip">
<p class="admonition-title">Jupyter Notebook: A Deep Dive into Dataset Structure</p>

Any user wishing to write custom code that works with robomimic datasets should also look at the [jupyter notebook](https://github.com/ARISE-Initiative/robomimic/blob/master/examples/notebooks/datasets.ipynb) at `examples/notebooks/datasets.ipynb`, which showcases several useful python code snippets for working with robomimic hdf5 datasets.

</div>

## Visualize Dataset Trajectories

<div class="admonition note">
<p class="admonition-title">Note: These examples are compatible with any robomimic dataset.</p>

The examples in this section use the small hdf5 dataset packaged with the repository in `tests/assets/test_v15.hdf5` (which requires robosuite v1.5.1), but you can run these examples with any robomimic hdf5 dataset.

</div>

Use the `playback_dataset.py` script to easily view dataset trajectories.

```sh
# For the first 5 trajectories, load environment simulator states one-by-one, and render "agentview" and "robot0_eye_in_hand" cameras to video at /tmp/playback_dataset.mp4
$ python playback_dataset.py --dataset ../../tests/assets/test_v15.hdf5 --render_image_names agentview robot0_eye_in_hand --video_path /tmp/playback_dataset.mp4 --n 5

# Directly visualize the image observations in the dataset. This is especially useful for real robot datasets where there is no simulator to use for rendering.
$ python playback_dataset.py --dataset ../../tests/assets/test_v15.hdf5 --use-obs --render_image_names agentview_image --video_path /tmp/obs_trajectory.mp4

# Visualize depth observations as well.
$ python playback_dataset.py --dataset /path/to/dataset.hdf5 --use-obs --render_image_names agentview_image --render_depth_names agentview_depth --video_path /tmp/obs_trajectory.mp4

# Play the dataset actions in the environment to verify that the recorded actions are reasonable.
$ python playback_dataset.py --dataset ../../tests/assets/test_v15.hdf5 --use-actions --render_image_names agentview --video_path /tmp/playback_dataset_with_actions.mp4

# Visualize only the initial demonstration frames.
$ python playback_dataset.py --dataset ../../tests/assets/test_v15.hdf5 --first --render_image_names agentview --video_path /tmp/dataset_task_inits.mp4
```
