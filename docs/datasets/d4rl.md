# D4RL

## Overview
The [D4RL](https://arxiv.org/abs/2004.07219) benchmark provides a set of locomotion tasks and demonstration datasets.

## Downloading

Use `convert_d4rl.py` in the `scripts/conversion` folder to automatically download and postprocess the D4RL dataset in a single step. For example:

```sh
# by default, download to robomimic/datasets
$ python convert_d4rl.py --env walker2d-medium-expert-v2
# download to specific folder
$ python convert_d4rl.py --env walker2d-medium-expert-v2 --folder /path/to/output/folder/
```

- `--env` specifies the dataset to download
- `--folder` specifies where you want to download the dataset. If no folder is provided, the `datasets` folder at the top-level of the repository will be used.

The script will download the raw hdf5 dataset to `--folder`, and the converted one that is compatible with this repository into the `converted` subfolder.

## Postprocessing

No postprocessing is required, assuming the above script is run!

## D4RL Results

Below, we provide a table of results on common D4RL datasets using the algorithms included in the released codebase. We follow the convention in the TD3-BC paper, where we average results over the final 10 rollout evaluations, but we use 50 rollouts instead of 10 for each evaluation. All results are reported on the `-v2` environment variants. Apart from a small handful of the halfcheetah results, the results align with those presented in the [TD3_BC paper](https://arxiv.org/abs/2106.06860). We suspect the halfcheetah results are different because we used `mujoco-py` version `2.1.2.14` in our evaluations, as opposed to `1.5` in order to be consistent with the version we were using for robosuite datasets. The results below were generated with `gym` version `0.24.1` and this `d4rl` [commit](https://github.com/Farama-Foundation/D4RL/tree/305676ebb2e26582d50c6518c8df39fd52dea587).

|                               | **BCQ**       | **CQL**       | **TD3-BC**    | **IQL**       |
| ----------------------------- | ------------- | ------------- | ------------- | ------------- |
| **HalfCheetah-Medium**        | 46.8% (5535)  | 46.7% (5516)  | 47.9% (5664)  | 45.6% (5379)  |
| **Hopper-Medium**             | 63.9% (2059)  | 59.2% (1908)  | 61.0% (1965)  | 53.7% (1729)  |
| **Walker2d-Medium**           | 74.6% (3426)  | 79.7% (3659)  | 82.9% (3806)  | 77.0% (3537)  |
| **HalfCheetah-Medium-Expert** | 89.9% (10875) | 77.6% (9358)  | 92.1% (11154) | 89.0% (10773) |
| **Hopper-Medium-Expert**      | 79.5% (2566)  | 62.9% (2027)  | 89.7% (2900)  | 110.1% (3564) |
| **Walker2d-Medium-Expert**    | 98.7% (4535)  | 109.0% (5007) | 111.1% (5103) | 109.7% (5037) |
| **HalfCheetah-Expert**        | 92.9% (11249) | 67.7% (8126)  | 94.6% (11469) | 93.3% (11304) |
| **Hopper-Expert**             | 92.3% (2984)  | 104.2% (3370) | 108.5% (3512) | 110.5% (3577) |
| **Walker2d-Expert**           | 108.6% (4987) | 108.5% (4983) | 110.3% (5066) | 109.1% (5008) |


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
