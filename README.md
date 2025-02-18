# robomimic

[**[Homepage]**](https://robomimic.github.io/) &ensp; [**[Documentation]**](https://robomimic.github.io/docs/introduction/overview.html) &ensp; [**[Study Paper]**](https://arxiv.org/abs/2108.03298) &ensp; [**[Study Website]**](https://robomimic.github.io/study/) &ensp; [**[ARISE Initiative]**](https://github.com/ARISE-Initiative)

-------
## Latest Updates
- [10/11/2023] **v0.3.1**: support for extracting, training on, and visualizing depth observations for robosuite datasets
- [07/03/2023] **v0.3.0**: BC-Transformer and IQL :brain:, support for DeepMind MuJoCo bindings :robot:, pre-trained image reps :eye:, wandb logging :chart_with_upwards_trend:, and more
- [05/23/2022] **v0.2.1**: Updated website and documentation to feature more tutorials :notebook_with_decorative_cover:
- [12/16/2021] **v0.2.0**: Modular observation modalities and encoders :wrench:, support for [MOMART](https://sites.google.com/view/il-for-mm/home) datasets :open_file_folder: [[release notes]](https://github.com/ARISE-Initiative/robomimic/releases/tag/v0.2.0) [[documentation]](https://robomimic.github.io/docs/v0.2/introduction/overview.html)
- [08/09/2021] **v0.1.0**: Initial code and paper release

-------

## Colab quickstart
Get started with a quick colab notebook demo of robomimic without installing anything locally.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1b62r_km9pP40fKF0cBdpdTO2P_2eIbC6?usp=sharing)


-------

**robomimic** is a framework for robot learning from demonstration.
It offers a broad set of demonstration datasets collected on robot manipulation domains and offline learning algorithms to learn from these datasets.
**robomimic** aims to make robot learning broadly *accessible* and *reproducible*, allowing researchers and practitioners to benchmark tasks and algorithms fairly and to develop the next generation of robot learning algorithms.

## Core Features

<p align="center">
  <img width="50.0%" src="docs/images/core_features.png">
 </p>

<!-- **Standardized Datasets**
- Simulated and real-world tasks
- Multiple environments and robots
- Diverse human-collected and machine-generated datasets

**Suite of Learning Algorithms**
- Imitation Learning algorithms (BC, BC-RNN, HBC)
- Offline RL algorithms (BCQ, CQL, IRIS, TD3-BC)

**Modular Design**
- Low-dim + Visuomotor policies
- Diverse network architectures
- Support for external datasets

**Flexible Workflow**
- Hyperparameter sweep tools
- Dataset visualization tools
- Generating new datasets -->


## Reproducing benchmarks

The robomimic framework also makes reproducing the results from different benchmarks and datasets easy. See the [datasets page](https://robomimic.github.io/docs/datasets/overview.html) for more information on downloading datasets and reproducing experiments.

## Troubleshooting

Please see the [troubleshooting](https://robomimic.github.io/docs/miscellaneous/troubleshooting.html) section for common fixes, or [submit an issue](https://github.com/ARISE-Initiative/robomimic/issues) on our github page.

## Contributing to robomimic
This project is part of the broader [Advancing Robot Intelligence through Simulated Environments (ARISE) Initiative](https://github.com/ARISE-Initiative), with the aim of lowering the barriers of entry for cutting-edge research at the intersection of AI and Robotics.
The project originally began development in late 2018 by researchers in the [Stanford Vision and Learning Lab](http://svl.stanford.edu/) (SVL).
Now it is actively maintained and used for robotics research projects across multiple labs.
We welcome community contributions to this project.
For details please check our [contributing guidelines](https://robomimic.github.io/docs/miscellaneous/contributing.html).

## Citation

Please cite [this paper](https://arxiv.org/abs/2108.03298) if you use this framework in your work:

```bibtex
@inproceedings{robomimic2021,
  title={What Matters in Learning from Offline Human Demonstrations for Robot Manipulation},
  author={Ajay Mandlekar and Danfei Xu and Josiah Wong and Soroush Nasiriany and Chen Wang and Rohun Kulkarni and Li Fei-Fei and Silvio Savarese and Yuke Zhu and Roberto Mart\'{i}n-Mart\'{i}n},
  booktitle={Conference on Robot Learning (CoRL)},
  year={2021}
}
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
