# robomimic

[**[Homepage]**](https://robomimic.github.io/) &ensp; [**[Documentation]**](https://robomimic.github.io/docs/introduction/overview.html) &ensp; [**[Study Paper]**](https://arxiv.org/abs/2108.03298) &ensp; [**[Study Website]**](https://robomimic.github.io/study/) &ensp; [**[ARISE Initiative]**](https://github.com/ARISE-Initiative)

-------
## Installation
1. Clone the repo with the `--recurse-submodules` flag.
2. (if applicable) switch to `r2d2` branch
3. Run `pip install -e .` in `robomimic`
4. Run `pip install -e .` in `robomimic/act/detr`

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
python robomimic/scripts/dataset_states_to_obs.py --camera_names <list-of-camera-names> --compress --exclude-next-obs --dataset <ds-path>
```
The `<list-of-camera-names>` depends on the environment you are using.
For example, for square: `agentview robot0_eye_in_hand` In some other cases: `robot0_agentview_left robot0_agentview_right robot0_eye_in_hand`
This script will generate a new dataset with the suffix `_im84.hdf5` in the same directory as `--dataset`

### LIBERO datasets
1. Install LIBERO in your virtual environment (would need to comment out robomimic and robosuite from LIBERO/requirements.txt if they are installed from source)
```
cd <path-to-libero-clone>
pip install -r requirements.txt
pip install -e .
```
In case pip complains about package version mismatches, please handle them manually. 

2. Download LIBERO datasets
```
python benchmark_scripts/download_libero_datasets.py --datasets=all --download-dir=<path-to-download-dir>
```
This will download datasets into 5 different directories namely `libero_goal`, `libero_spatial`, `libero_object`, `libero_90`, `libero_10`.

-------
## Training
There are a number of algorithms to choose from: Diffusion Policy, ACT, BC-Transformer, etc.

Each algorithm has its own config generator script. For example for diffusion policy run:
```
python robomimic/scripts/config_gen/diffusion_gen.py --name <run-name>
```
You can add `--debug` to generate small runs for testing. Running this script will generate training run commands. You can use this script for generating a single run or multiple (for comparing settings and hyperparameter tuning).
After running this script you just need to run the command(s) outputted.

Want to learn how to set your own config values and sweep them? Read this short [tutorial section](https://robomimic.github.io/docs/tutorials/hyperparam_scan.html#step-3-set-hyperparameter-values).

-------
## Evaluation
In order to efficiently run evaluation, we have split up the training and eval scripts.  Our suggestion is to run train.py first, and then run multitask_eval.py once a sufficient number of checkpoints have built up.  When working on a cluster, you can launch multiple sbatch runs of multitask_eval.py and they will each eval different checkpoints and not overwrite eachother's progress.  Call multitask_eval.py as follows:
```
python robomimic/scripts/multitask_eval.py --train_dir path/to/training/run/directory --num_episodes 50
```

Num episodes is the number of rollouts you are running for each task.  Tasks are taken from config.train.data in the config file that is located in --train_dir.  You can alter this config if you want to change the tasks that are evaluated. There are a number of other parameters you can alter as well, so refer to the multitask_eval documentation for more information.

### Loading model checkpoint weights
Want to intialize your model with weights from a previous model checkpoint? Set the checkpoint path under `experiment.ckpt_path` in the config.

-------
## Logging and viewing results
Read this short [tutorial page](https://robomimic.github.io/docs/tutorials/viewing_results.html).

-------
## Real robot evaluation (r2d2 only)
Use this forked r2d2: https://github.com/snasiriany/r2d2/tree/robomimic-eval. Note that the branch is `robomimic-eval`.

Run this script: https://github.com/snasiriany/r2d2/blob/robomimic-eval/scripts/evaluation/evaluate_policy.py. Before doing so, make sure to fill out `CKPT_PATH` with the path to the saved robomimic checkpoint you wish to evaluate.
