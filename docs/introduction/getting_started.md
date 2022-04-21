# Getting Started

## Running experiments
We begin with a quick tutorial on downloading datasets and running experiments.

Before beginning, make sure you are at the base repo path:
```sh
$ cd {/base/repo/path}
```

### Step 1: Download dataset

Download the robosuite lift proficient human dataset:
```sh
$ python robomimic/scripts/download_datasets.py --tasks lift --dataset_types ph
```

The dataset can be found at `datasets/lift/ph/low_dim.hdf5`

### Step 2: Launch experiment

Now, we will run an experiment using `train.py`. In this case we would like to run behavior cloning (BC) for the lift dataset we just downloaded. We add the flag `--debug` to run a quick sample experiment:

```sh
$ python robomimic/scripts/train.py --config robomimic/exps/templates/bc.json --dataset datasets/lift/ph/low_dim.hdf5 --debug
```

<div class="admonition warning">
<p class="admonition-title">Warning!</p>

This example [requires robosuite](./installation.html#robosuite) to be installed (under the `offline_study` branch), but it can be run without robosuite by disabling rollouts in `robomimic/exps/templates/bc.json`: simply change the `experiment.rollout.enabled` flag to `false`.

</div>

### Step 3: View experiment results

After the script finishes, we can check the training outputs in the directory `bc_trained_models/test`.
Experiment outputs comprise the following:
```
config.json               # config used for this experiment
logs/                     # experiment log files
  log.txt                    # terminal output
  tb/                        # tensorboard logs
videos/                   # videos of robot rollouts during training
models/                   # saved model checkpoints
```

The experiment results can be viewed using tensorboard:
```sh
$ tensorboard --logdir bc_trained_models/test --bind_all
```

## Next steps
<!--
High-level overview of the `robomimic` directory (highlighting selected files):
```
algo/                     # algorithms
  bc.py                      # bc implementation
  ...
config/                   # default algorithm configs
  bc_config.py               # default config for bc
  ...
envs/                     # environment wrappers
  ...
exps/                     # custom experiment configs (overriding default algorithm configs)
  templates/                 # template experiment configs
    bc.json                     # template config for bc
    ...
models/                   # network architectures
  ...
scripts/                  # scripts
  train.py                   # main script for running experiments
  download_datasets.py       # downloading robomimic v0.1 datasets
  playback_dataset.py        # visualizing dataset trajectories
  ...
utils/                    # utils for training, evaluation, visualization, hp sweeps, etc
  ...
```
-->
For next steps, please refer to the remaining documentation sections: 
- [Datasets](../datasets/overview.html): downloading and benchmarking datasets
- [Tutorials](../tutorials/understanding_dataset_contents.html): creating datasets and algorithms, and running experiments
- [Modules](../modules/overview.html): detailed information on robomimic modules, eg. algorithms, architectures, experiments configs, etc
- [Troubleshooting](../miscellaneous/troubleshooting.html) and [Contributing guidelines](../miscellaneous/contributing.html)