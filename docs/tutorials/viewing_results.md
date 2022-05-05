# Logging and Viewing Training Results

In this section, we describe how to configure the logging and evaluations that occur during your training run, and how to view the results of a training run.

## Configuring Logging

TODO: describe different parts of config for logging (e.g. log terminal to text), setting whether rollouts happen and frequency, saving videos

### Saving Experiment Logs 
Configured under `experiment.logging`:
- **Saving terminal outputs**: set `terminal_output_to_txt` to `true`. Saved logs are located in `logs/log.txt` under the experiment folder. 
- **Saving logs to tensorboard**: set `logging.log_tb` to `true`. Saved logs are located in `logs/tb` under the experiment folder. 

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

TODO: put some information from https://arise-initiative.github.io/robomimic-web/docs/introduction/quickstart.html#logs-models-and-rollout-videos 

TODO: output directory structure (like in getting started, but more detail)

TODO: describe how to view tensorboard, and important sections of interest (rollout success rate, training loss)


<div class="admonition tip">
<p class="admonition-title">Loading Trained Checkpoints</p>

Please see the [Using Pretrained Models](./using_pretrained_models.html) tutorial to see how to load the trained model checkpoints in the `models` directory.

</div>