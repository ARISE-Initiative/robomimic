# RoboCasa Policy Learning Repo

This is the official policy learning repo accompanying the [RoboCasa](https://robocasa.ai/) release. This repo is based on top of robomimic, with modifications to train on RoboCasa datasets.

-------
## Installation
After installing the [RoboCasa codebase](https://github.com/robocasa/robocasa), follow the instructions below:
```
git clone https://github.com/ARISE-Initiative/robomimic -b robocasa
cd robomimic
pip install -e .
```

-------
## Training
There are a number of algorithms to choose from. We offer official support for BC-Transformer. Users can also adapt the code to run Diffusion Policy, ACT, etc.

Before training, download datasets, see instructions [here](https://github.com/robocasa/robocasa?tab=readme-ov-file#datasets).

Each algorithm has its own config generator script. For example for BC-Transformer policy run:
```
python robomimic/scripts/config_gen/bc_xfmr.py --name <experiment-name>
```
Modify this file accordingly, depending on which datasets you are training on and whether you are running evaluations.

Note: You can add `--debug` to generate small runs for testing.

Running this script will generate training run commands. You can use this script for generating a single run or multiple (for comparing settings and hyperparameter tuning).
After running this script you just need to run the command(s) outputted.

Want to learn how to set your own config values and sweep them? Read this short [tutorial section](https://robomimic.github.io/docs/tutorials/hyperparam_scan.html#step-3-set-hyperparameter-values).

### Loading model checkpoint weights
Want to intialize your model with weights from a previous model checkpoint? Set the checkpoint path under `experiment.ckpt_path` in the config.

### Logging and viewing results
Read this short [tutorial page](https://robomimic.github.io/docs/tutorials/viewing_results.html).

-------
## Training on new datasets
Before training, you need to pre-process your datasets to ensure they're the correct format.

1. Convert the raw robosuite dataset to robomimic format
```
python robomimic/scripts/conversion/convert_robosuite.py --dataset <ds-path>
```

This script will extract actions and add filter keys.

2. Extract image observations from robomimic dataset
```
OMP_NUM_THREADS=1 MPI_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python robomimic/scripts/dataset_states_to_obs.py --dataset <ds-path>
```
This script will generate a new dataset with the suffix `_im128.hdf5` in the same directory as `--dataset`

Note: you can add the flag `--generative_textures` to render images with AI-generated environment textures, and `--randomize_cameras` to randomize camera viewpoints for rendering.
