# Understanding Dataset Contents

This tutorial shows some easy ways to understand the contents of each hdf5 dataset.

<div class="admonition tip">
<p class="admonition-title">Jupyter Notebook: A Deep Dive into Dataset Structure</p>

The rest of this tutorial shows how to use utility scripts to work with robomimic datasets. While this should suffice for most users, any user wishing to write custom code that works with robomimic datasets should look at the jupyter notebook at `examples/notebooks/datasets.ipynb`, which showcases several useful python code snippets for working with robomimic hdf5 datasets. These code snippets are used under-the-hood in our codebase. 

</div>

## View Dataset Structure and Videos

<div class="admonition note">
<p class="admonition-title">Note: These examples are compatible with any robomimic dataset.</p>

The examples in this section use the small hdf5 dataset packaged with the repository in `tests/assets/test.hdf5`, but you can run these examples with any robomimic hdf5 dataset. If you are using the default dataset, please make sure that robosuite is on the `offline_study` branch of robosuite -- this is necessary for the playback scripts to function properly.

</div>

The repository offers a simple utility script (`get_dataset_info.py`) to view the hdf5 dataset structure and some statistics of hdf5 datasets. The script will print out some statistics about the trajectories, the filter keys present in the dataset, the environment metadata in the dataset, and the dataset structure for the first demonstration. Pass the `--verbose` argument to print the list of demonstration keys under each filter key, and the dataset structure for all demonstrations.

```sh
$ python get_dataset_info.py --dataset ../../tests/assets/test.hdf5
```

The repository also offers a utility script (`playback_dataset.py`) that allows you to easily view dataset trajectories, and verify that the recorded dataset actions are reasonable. The example below loads the saved MuJoCo simulator states one by one in a simulation environment, and renders frames from some simulation cameras to generate a video, for the first 5 trajectories. This is an easy way to view trajectories from the dataset. After this script runs, you can view the video at `/tmp/playback_dataset.mp4`.

```sh
$ python playback_dataset.py --dataset ../../tests/assets/test.hdf5 --render_image_names agentview robot0_eye_in_hand --video_path /tmp/playback_dataset.mp4 --n 5
```

An alternative way to view the demonstrations is to directly visualize the image observations in the dataset. This is especially useful for real robot datasets, where there is no simulator to use for rendering.

```sh
$ python playback_dataset.py --dataset ../../tests/assets/test.hdf5 --use-obs --render_image_names agentview_image --video_path /tmp/obs_trajectory.mp4
```

It's also easy to use the script to verify that the dataset actions are reasonable, by playing the actions back one by one in the environment.

```sh
$ python playback_dataset.py --dataset ../../tests/assets/test.hdf5 --use-actions --render_image_names agentview --video_path /tmp/playback_dataset_with_actions.mp4
```

Finally, the script can be used to visualize the initial states in the demonstration data.

```sh
$ python playback_dataset.py --dataset ../../tests/assets/test.hdf5 --first --render_image_names agentview --video_path /tmp/dataset_task_inits.mp4
```
