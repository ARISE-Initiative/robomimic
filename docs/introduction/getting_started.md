# Getting Started

## Quickstart colab notebook

If you prefer to test the basic functionality of robomimic without installing anything locally, try the quickstart [Colab notebook](https://colab.research.google.com/drive/1b62r_km9pP40fKF0cBdpdTO2P_2eIbC6?usp=sharing).

## Running experiments
We begin with a quick tutorial on downloading datasets and running experiments.

Before beginning, make sure you are at the base repo path:
```sh
$ cd {/path/to/robomimic}
```

### Step 1: Download dataset

Download the robosuite **Lift (PH)** dataset (see [this link](../datasets/robomimic_v0.1.html#proficient-human-ph) for more information on this dataset):
```sh
$ python robomimic/scripts/download_datasets.py --tasks lift --dataset_types ph
```

The dataset can be found at `datasets/lift/ph/low_dim_v141.hdf5`

### Step 2: Launch experiment

Now, we will run an experiment using `train.py`. In this case we would like to run behavior cloning (BC) for the lift dataset we just downloaded.

```sh
$ python robomimic/scripts/train.py --config robomimic/exps/templates/bc.json --dataset datasets/lift/ph/low_dim_v141.hdf5 --debug
```

<div class="admonition note">
<p class="admonition-title">Running quick sanity check experiments</p>

Make sure to add the `--debug` flag to your experiments as a sanity check that your implementation works.

</div>

**Resume functionality**: If your training job fails due to any reason, you can re-launch your job with the additional `--resume` flag to resume training from the last saved epoch. This will resume training from the `last.pth` checkpoint in your output directory for the most recent training run. Some points to note:
1. While fine-tuning from a specified checkpoint (in `config.experiment.ckpt_path`) would load model weights from the checkpoint, resume functionality also loads the optimizer state.
2. `config.experiment.ckpt_path` will be ignored if you are resuming a training job, i.e. `last.pth` will take precedence if `--resume` is passed.

<div class="admonition warning">
<p class="admonition-title">Warning!</p>

This example [requires robosuite](./installation.html#install-simulators) to be installed (under the `v1.5.1` branch), but it can be run without robosuite by disabling rollouts in `robomimic/exps/templates/bc.json`: simply change the `experiment.rollout.enabled` flag to `false`.

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
Please refer to the remaining documentation sections. Some helpful suggestions on pages to view next:
- [Configuring and Launching Training Runs](../tutorials/configs.html)
- [Logging and Viewing Training Results](../tutorials/viewing_results.html)
- [Running Hyperparameter Scans](../tutorials/hyperparam_scan.html)
- [Overview of Datasets](../datasets/overview.html)
- [Dataset Contents and Visualization](../tutorials/dataset_contents.html)
- [Overview of Modules](../modules/overview.html)
