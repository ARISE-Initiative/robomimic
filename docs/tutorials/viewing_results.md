# Logging and Viewing Training Results

In this section, we describe how to configure the logging and evaluations that occur during your training run, and how to view the results of a training run.

## Configuring Logging

### Saving Experiment Logs 
Configured under `experiment.logging`:
- **Saving terminal outputs**: set `terminal_output_to_txt` to `true`. Saved logs are located in `logs/log.txt` under the experiment folder. 
- **Saving logs to tensorboard**: set `logging.log_tb` to `true`. Saved logs are located in `logs/tb/` under the experiment folder. 

### Saving Model Checkpoints
Configured under `experiment.save`:
- **Enable saving model checkpoints**: set `enabled` to `true`
- **Control frequency of checkpoints**: `every_n_epochs`, `every_n_seconds`, `epochs` (list)
- **Save best checkpoints**: `on_best_validation`, `on_best_rollout_return` `on_best_rollout_success_rate`

### Evaluating Rollouts and Saving Videos
#### Evaluating Rollouts
Configured under `experiment.rollout`:
- **Enable evaluation rollouts**: set `enabled` to `true`
- **Number of rollouts per evaluation**: `n`
- **Rollout horizon**: `horizon`
- **Frequency of evaluation (in epochs)**: `rate`
- **Terminating rollouts on task success**: `terminate_on_success`
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
Experiment results (y-axis) are logged across epochs (x-axis).
You may find the following logging metrics useful:
- `Rollout/`: evaluation rollout metrics, eg. success rate, rewards, etc.
- `Timing_Stats/`: time spent by the algorithm loading data, training, performing rollouts, etc.
- `Timing_Stats/`: time spent by the algorithm loading data, training, performing rollouts, etc.
- `Train/`: training stats
- `Validation/`: validation stats
- `System/RAM Usage (MB)`: system RAM used by algorithm