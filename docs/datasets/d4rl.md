# D4RL

## Overview
The [D4RL](https://arxiv.org/abs/2004.07219) benchmark set of tasks provide a set of locomotion tasks and collected demonstration data.

## Downloading

Use `convert_d4rl.py` in the `scripts/conversion` folder to automatically download and postprocess the D4RL dataset in a single step. For example:

```sh
# by default, download to robomimic/datasets
$ python convert_d4rl.py --env walker2d-medium-expert-v0
# download to specific folder
$ python convert_d4rl.py --env walker2d-medium-expert-v0 --folder /path/to/output/folder/
```

- `--env` specifies the dataset to download
- `--folder` specifies where you want to download the dataset. If no folder is provided, the `datasets` folder at the top-level of the repository will be used.

The script will download the raw hdf5 dataset to `--folder`, and the converted one that is compatible with this repository into the `converted` subfolder.

## Postprocessing

No postprocessing is required, assuming the above script is run!

## D4RL Results

Below, we provide a table of results on common D4RL datasets using the algorithms included in the released codebase. We follow the convention in the TD3-BC paper, where we average results over the final 10 rollout evaluations, but we use 50 rollouts instead of 10 for each evaluation. Apart from a small handful of the halfcheetah results, the results align with those presented in the [TD3_BC paper](https://arxiv.org/abs/2106.06860). We suspect the halfcheetah results are different because we used `mujoco-py` version `2.0.2.13` in our evaluations, as opposed to `1.5` in order to be consistent with the version we were using for robosuite datasets. The results below were generated with `gym` version `0.17.3` and this `d4rl` [commit](https://github.com/rail-berkeley/d4rl/tree/9b68f31bab6a8546edfb28ff0bd9d5916c62fd1f).

|                               | **BCQ**       | **CQL**       | **TD3-BC**    |
| ----------------------------- | ------------- | ------------- | ------------- |
| **HalfCheetah-Medium**        | 40.8% (4791)  | 38.5% (4497)  | 41.7% (4902)  |
| **Hopper-Medium**             | 36.9% (1181)  | 30.7% (980)   | 97.9% (3167)  |
| **Walker2d-Medium**           | 66.4% (3050)  | 65.2% (2996)  | 77.0% (3537)  |
| **HalfCheetah-Medium-Expert** | 74.9% (9016)  | 21.5% (2389)  | 79.4% (9578)  |
| **Hopper-Medium-Expert**      | 83.8% (2708)  | 111.7% (3614) | 112.2% (3631) |
| **Walker2d-Medium-Expert**    | 70.2% (3224)  | 77.4% (3554)  | 102.0% (4683) |
| **HalfCheetah-Expert**        | 94.3% (11427) | 29.2% (3342)  | 95.4% (11569) |
| **Hopper-Expert**             | 104.7% (3389) | 111.8% (3619) | 112.2% (3633) |
| **Walker2d-Expert**           | 80.5% (3699)  | 108.0% (4958) | 105.3% (4837) |


### Reproducing D4RL Results

In order to reproduce the results above, first make sure that the `generate_paper_configs.py` script has been run, where the `--dataset_dir` argument is consistent with the folder where the D4RL datasets were downloaded using the `convert_d4rl.py` script. This is also the first step for reproducing results on the released robot manipulation datasets. The `--config_dir` directory used in the script (`robomimic/exps/paper` by default) will contain a `d4rl.sh` script, and a `d4rl` subdirectory that contains all the json configs. The table results above can be generated simply by running the training commands in the shell script.

## Citation
```sh
@article{fu2020d4rl,
  title={D4rl: Datasets for deep data-driven reinforcement learning},
  author={Fu, Justin and Kumar, Aviral and Nachum, Ofir and Tucker, George and Levine, Sergey},
  journal={arXiv preprint arXiv:2004.07219},
  year={2020}
}
```