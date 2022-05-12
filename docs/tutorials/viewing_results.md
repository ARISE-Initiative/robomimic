# Logging and Viewing Training Results

In this section, we describe how to configure the logging and evaluations that occur during your training run, and how to view the results of a training run.

## Configuring Logging

### Saving Experiment Logs 
Configured under `experiment.logging`:
```
# save terminal outputs under `logs/log.txt` in experiment folder
terminal_output_to_txt (bool)

# save tensorboard logs under `logs/tb` in experiment folder
log_tb (bool)
```

### Saving Model Checkpoints
Configured under `experiment.save`:
```
# enable saving model checkpoints
enabled (bool)

# controlling frequency of checkpoints
every_n_epochs (int)
every_n_seconds (int)
epochs (list)

# saving the best checkpoints
on_best_validation (bool)
on_best_rollout_return (bool)
on_best_rollout_success_rate (bool)
```

### Evaluating Rollouts and Saving Videos
#### Evaluating Rollouts
Configured under `experiment.rollout`:
```
enabled (bool)                # enable evaluation rollouts

n (int)                       # number of rollouts per evaluation
horizon (int)                 # number of timesteps per rollout
rate (int)                    # frequency of evaluation (in epochs)
terminate_on_success (bool)   # terminating rollouts upon task success
```

#### Saving Videos
To save videos of the rollouts, set `experiment.render_video` to `true`.

## Viewing Training Results

### Contents of Training Outputs
After the script finishes, you can check the training outputs in the `<train.output_dir>/<experiment.name>/<date>` experiment directory:
```
config.json               # config used for this experiment
logs/                     # experiment log files
  log.txt                    # terminal output
  tb/                        # tensorboard logs
videos/                   # videos of robot rollouts during training
models/                   # saved model checkpoints
```

<div class="admonition tip">
<p class="admonition-title">Loading Trained Checkpoints</p>

Please see the [Using Pretrained Models](./using_pretrained_models.html) tutorial to see how to load the trained model checkpoints in the `models` directory.

</div>

### Viewing Tensorboard Results
The experiment results can be viewed using tensorboard:
```sh
$ tensorboard --logdir <experiment-log-dir> --bind_all
```
Below is a snapshot of the tensorboard dashboard:
<p align="center">
  <img width="99.0%" src="../images/tensorboard.png">
</p>

Experiment results (y-axis) are logged across epochs (x-axis).
You may find the following logging metrics useful:
- `Rollout/`: evaluation rollout metrics, eg. success rate, rewards, etc.
- `Timing_Stats/`: time spent by the algorithm loading data, training, performing rollouts, etc.
- `Timing_Stats/`: time spent by the algorithm loading data, training, performing rollouts, etc.
- `Train/`: training stats
- `Validation/`: validation stats
- `System/RAM Usage (MB)`: system RAM used by algorithm