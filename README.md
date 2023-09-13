# robomimic

[**[Homepage]**](https://robomimic.github.io/) &ensp; [**[Documentation]**](https://robomimic.github.io/docs/introduction/overview.html) &ensp; [**[Study Paper]**](https://arxiv.org/abs/2108.03298) &ensp; [**[Study Website]**](https://robomimic.github.io/study/) &ensp; [**[ARISE Initiative]**](https://github.com/ARISE-Initiative)

-------
## Installation
Clone this repo (make sure you are on the right branch). Then run `pip install -e .` in the base repo path.

-------
## Pre-processing datasets
Before training, you need to pre-process your datasets to ensure they're the correct format.
### r2d2 datasets
Convert the raw r2d2 data using this simple script:
```
python robomimic/scripts/convertion/convert_r2d2.py --folder <folder-containing-r2d2-data>
```
### robosuite datasets
1. Convert the raw robosuite dataset to robomimic format
```
python robomimic/scripts/conversion/convert_robosuite.py --dataset <ds-path> --filter_num_demos <list-of-numbers>
```
`--filter_num_demos` corresponds to the number of demos to filter by. It's a list, eg. `10 30 50 100 200 500 1000`

This script will extract absolute actions, extract the action dict, and add filter keys.

2. Extract image observations from robomimic dataset
```
python robomimic/scripts/dataset_states_to_obs.py --camera_names robot0_agentview_left robot0_agentview_right robot0_eye_in_hand --compress --exclude-next-obs --dataset <ds-path>
```
This script will generate a new dataset with the suffix `_im84.hdf5` in the same directory as `--dataset`

-------
## Training
There are a number of algorithms to choose from: Diffusion Policy, ACT, BC-Transformer, etc.

Each algorithm has its own config generator script. For example for diffusion policy run:
```
python robomimic/scripts/config_gen/diffusion_gen.py --name <run-name>
```
You can add `--debug` to generate small runs for testing. Running this script will generate training run commands. You can use this script for generating a single run or multiple (for comparing settings and hyperparameter tuning).
At this point you just need to run the command(s) outputted.

### Loading model checkpoint weights
Want to intialize your model with weights from a previous model checkpoint? Set the checkpoint path under `experiment.ckpt_path` in the config.

-------
## Logging and viewing results
Read this short [tutorial page](https://robomimic.github.io/docs/tutorials/viewing_results.html).

-------
## Real robot evaluation (r2d2 only)
Use this forked r2d2: https://github.com/snasiriany/r2d2/tree/robomimic-eval. Note that the branch is `robomimic-eval`.

Run this script: https://github.com/snasiriany/r2d2/blob/robomimic-eval/scripts/evaluation/evaluate_policy.py. Before doing so, make sure to fill out `CKPT_PATH` with the path to the saved robomimic checkpoint you wish to evaluate.
