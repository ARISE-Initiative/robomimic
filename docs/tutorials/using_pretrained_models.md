# Using Pretrained Models

This tutorial shows how to use pretrained model checkpoints.

<div class="admonition tip">
<p class="admonition-title">Jupyter Notebook: Working with Pretrained Policies</p>

The rest of this tutorial shows how to use utility scripts to load and rollout a trained policy. If you wish to do so via an interactive notebook, please refer to the [jupyter notebook](https://github.com/ARISE-Initiative/robomimic/blob/master/examples/notebooks/run_policy.ipynb) at `examples/notebooks/run_policy.ipynb`. The notebook tutorial shows how to download a checkpoint from the model zoo, load the checkpoint in pytorch, and rollout the policy. 

</div>

## Evaluating Trained Policies

Saved policy checkpoints in the `models` directory can be evaluated using the `run_trained_agent.py` script:
```sh
# 50 rollouts with max horizon 400 and render agentview and wrist camera images to video
$ python run_trained_agent.py --agent /path/to/model.pth --n_rollouts 50 --horizon 400 --seed 0 --video_path /path/to/output.mp4 --camera_names agentview robot0_eye_in_hand 

# Write rollouts to a new dataset hdf5
python run_trained_agent.py --agent /path/to/model.pth --n_rollouts 50 --horizon 400 --seed 0 --dataset_path /path/to/output.hdf5 --dataset_obs

# Write rollouts without explicit observations to hdf5
python run_trained_agent.py --agent /path/to/model.pth --n_rollouts 50 --horizon 400 --seed 0 --dataset_path /path/to/output.hdf5
```

In the last case, the observations can be (later) extracted later using the `dataset_states_to_obs.py` script (see [here](../datasets/robosuite.html#extracting-observations-from-mujoco-states)).
