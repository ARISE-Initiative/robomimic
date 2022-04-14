# Using Pretrained Models

This tutorial shows how to use pretrained model checkpoints.

TODO: Danfei, check with Soroush on information in new introduction section, and figure out what stuff needs to go in this tutorial.

## Evaluating Trained Policies

Saved policy checkpoints in the `models` directory can be evaluated using the `run_trained_agent.py` script. The below example can be used to evaluate a policy with 50 rollouts of maximum horizon 400 and save the rollouts to a video. The agentview and wrist camera images are used to render video frames.

```sh
$ python run_trained_agent.py --agent /path/to/model.pth --n_rollouts 50 --horizon 400 --seed 0 --video_path /path/to/output.mp4 --camera_names agentview robot0_eye_in_hand 
```

The 50 agent rollouts can also be written to a new dataset hdf5.

```sh
python run_trained_agent.py --agent /path/to/model.pth --n_rollouts 50 --horizon 400 --seed 0 --dataset_path /path/to/output.hdf5 --dataset_obs 
```

Instead of storing the observations, which can consist of high-dimensional images, they can be excluded by omitting the `--dataset_obs` flag. The observations can be extracted using the `dataset_states_to_obs.hdf5` script (see the Datasets documentation for more information on this).

```sh
python run_trained_agent.py --agent /path/to/model.pth --n_rollouts 50 --horizon 400 --seed 0 --dataset_path /path/to/output.hdf5
```
